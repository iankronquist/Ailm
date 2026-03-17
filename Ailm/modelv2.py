'''
This model starts from the foundation developed for AilmV1 and experiments with modifications to the residual stream.
Specifically, we're looking at evaluating Manifold-Constrained Hyper-Connections, and possibly a design similar to ResFormer.

Value Residual Learning: https://arxiv.org/pdf/2410.17897
Other Value residual technique: https://github.com/KellerJordan/modded-nanogpt/tree/new_record/records/110624_ShortcutsTweaks

HYPER-CONNECTIONS: https://arxiv.org/pdf/2409.19606
mHC: Manifold-Constrained Hyper-Connections: https://arxiv.org/pdf/2512.24880


'''

import dataclasses
import math
import typing

import mlx
from mlx import nn
from mlx.core import array as Tensor
import mlx.core

from model import AilmV1, AilmV1Config, Mlp, _init_weights


@dataclasses.dataclass
class AilmV2Config(AilmV1Config):
    '''
    Extends AilmV1Config
    '''

    # Set to 1 for normal operation, and say 4 for mHC like behavior.
    hyper_connection_residual_stream_expansion_factor: int = 1
    use_value_residual_technique: bool = False

    residual_size: int = dataclasses.field(init=False, repr=False) 

    def __post_init__(self):
        super().__post_init__()
        self.residual_size = self.hidden_size * self.hyper_connection_residual_stream_expansion_factor


class CausalSelfAttentionV2(nn.Module):
    '''
    CausalSelfAttention
    
    We use Grouped Query Attention to save parameters: https://arxiv.org/pdf/2305.13245
    '''
    def __init__(self, config: AilmV2Config, rope: nn.RoPE):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.head_count = config.head_count

        self.key_value_head_count = config.key_value_head_count
        self.head_dim = config.hidden_size // config.head_count

        # Initialize the Wk, Wq, Wv linear projections which are all packed together in one big matrix so we can dispatch on big matrix multiply to the GPU instead of three smaller ones.
        out_dim = self.hidden_size + 2 * self.head_dim * self.key_value_head_count
        self.c_attn = nn.Linear(self.hidden_size, out_dim, bias=False)

        # Initialize the output projection back to residual space.
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.c_proj.set_dtype(config.dtype)
        self.c_attn.set_dtype(config.dtype)

        # Empty initial cache.
        self.k_cache = None
        self.v_cache = None
        self.use_kv_cache = config.use_kv_cache

        # Hold onto the shared rope module.
        self.rope = rope

        # Calculate our scale for our attention step.
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Calculate the size of the Q, K, and V vectors so we can slice them out of the packed result.
        self.q_size = self.head_count * self.head_dim
        self.kv_size = self.key_value_head_count * self.head_dim

        # See "Attention Is Off By One": https://www.evanmiller.org/attention-is-off-by-one.html
        if config.use_attention_sinks:
            self.sinks = mlx.core.zeros((self.head_count,), config.dtype)
        else:
            self.sinks = None

        self.use_value_residual_technique = config.use_value_residual_technique
        self.v1_lambda = mlx.core.array(0.5, dtype=config.dtype)


    def __call__(self, x: Tensor, first_layer_value: typing.Optional[Tensor]) -> typing.Tuple[Tensor, typing.Optional[Tensor]]:
        # We expect this hidden tensor x to have the dimensions for the Batch length, sequence length (T), and Channels (hidden_size)
        B, T, C = x.shape

        # The Wq, Wk, an Wv weights are all packed together so we can dispatch on big matrix multiply to the GPU instead of three smaller ones.
        qkv = self.c_attn(x)

        # Slice the queries, keys, and values out of the packed result.
        q = qkv[:, :, :self.q_size]
        k = qkv[:, :, self.q_size:(self.q_size+self.kv_size)]
        v = qkv[:, :, (self.q_size+self.kv_size):]

        # Value residual learning.
        if self.use_value_residual_technique:
            if first_layer_value is None:
                # We must be the first layer. Pass our v on directly to future layers.
                first_layer_value = v
            else:
                # I think of this as each layer updating the first layer's value vector.
                # v1_lambda is the amount of the original value vector we preserve.
                v = (1. - self.v1_lambda) * v + self.v1_lambda * first_layer_value

        # Reshape them to the layout we expect for rope and group query attention.
        q = q.reshape(B, T, self.head_count,  self.head_dim)
        q = mlx.core.swapaxes(q, 1, 2)
        k = k.reshape(B, T, self.key_value_head_count, self.head_dim)
        k = mlx.core.swapaxes(k, 1, 2)
        v = v.reshape(B, T, self.key_value_head_count, self.head_dim)
        v = mlx.core.swapaxes(v, 1, 2)

        if self.use_kv_cache and self.k_cache is not None:
            # If we have a cache, the new tokens should have positions starting after the cache
            offset = self.k_cache.shape[2]  # Cache shape is (B, H, Seq, D)
        else:
            offset = 0

        # Apply Rotational Positional Embeddings.
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # Extract cached entries if necessary.
        if self.use_kv_cache:
            if self.k_cache is None or self.v_cache is None:
                # If there is nothing cached yet, k and v make up the whole cache.
                self.k_cache = k
                self.v_cache = v
            else:
                # If there is something cached already, append the arrays to the cache.
                self.k_cache = mlx.core.concatenate([self.k_cache, k], axis=2)
                self.v_cache = mlx.core.concatenate([self.v_cache, v], axis=2)
            # Use the cached context.
            k = self.k_cache
            v = self.v_cache

        # Scaled dot product attention with causal masking.
        if self.use_kv_cache:
            # If we are using KV caching, and the length of our input sequence is 1, 
            # we can skip the causal mask as a performance optimization.
            mask = "causal" if T > 1 else None
        else:
            mask = "causal"
        y = mlx.core.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask, sinks=self.sinks)

        # Reshape the tensor so it's laid out like the original residual stream.
        y = y.transpose(0, 2, 1, 3).reshape([B, T, C])

        # Project back to the residual stream space.
        y = self.c_proj(y)

        return y, first_layer_value

    def reset_key_value_cache(self, enable_key_value_cache: bool):
        '''
        Reset the Key Value Cache for this attention module.
        If enable_key_value_cache is True, future calls to this module will cache previous inference values, which is recommended during inference, but not during training.
        Otherwise, if enable_key_value_cache is False, no caching will occur.
        '''
        self.k_cache = None
        self.v_cache = None
        self.use_kv_cache = enable_key_value_cache


def rms_norm(hidden: Tensor) -> Tensor:
    return mlx.core.fast.rms_norm(hidden, None, eps=1e-8)

def identity(hidden: Tensor) -> Tensor:
    return hidden


class Mixing(nn.Module):


    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mlx.core.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims),
        )
        self.bias = mlx.core.random.uniform(
                low=-scale,
                high=scale,
                shape=(output_dims,),
            )
        self.alpha = mlx.core.random.uniform(low=-scale, high=scale, shape=(1,))

        self.gate = mlx.core.tanh

    def __call__(self, x: Tensor) -> Tensor:
        # 3. Preliminary
        # https://arxiv.org/pdf/2512.24880
        return self.alpha * self.gate(self.weight @ x.T) + self.bias

class HyperConnection(nn.Module):
    def __init__(self, config: AilmV2Config, inner_block: nn.Module):
            residual_size = config.residual_size

            self.layer_norm = rms_norm

            self.h_res  = Mixing(residual_size, residual_size)
            self.h_pre  = Mixing(residual_size, config.hidden_size)
            self.h_post = Mixing(config.hidden_size, residual_size)

            self.inner = inner_block

    def __call__(self, residual: Tensor) -> Tensor:
        residual = self.h_res(residual) + self.h_post(self.inner(self.h_pre(self.layer_norm(residual))))
        return residual

class LayerV2(nn.Module):
    '''
    A single layer in the model.
    '''
    def __init__(self, config: AilmV2Config, rope: nn.RoPE):
        '''
        Initialize a layer. The argument rope must be a nn.RoPE module.
        '''
        super().__init__()

        self.attn = CausalSelfAttentionV2(config, rope)
        self.ln_1 = rms_norm
        self.mlp = Mlp(config)
        self.ln_2 = rms_norm

        # TODO: What if we added hyper connections around MLP & Attention blocks?
        if config.hyper_connection_residual_stream_expansion_factor == 1:
            self.h1_res  = identity
            self.h1_pre  = identity
            self.h1_post = identity

            self.h2_res  = identity
            self.h2_pre  = identity
            self.h2_post = identity
        else:
            residual_size = config.residual_size

            self.h1_res  = Mixing(residual_size, residual_size)
            self.h1_pre  = Mixing(residual_size, config.hidden_size)
            self.h1_post = Mixing(config.hidden_size, residual_size)

            self.h2_res  = Mixing(residual_size, residual_size)
            self.h2_pre  = Mixing(residual_size, config.hidden_size)
            self.h2_post = Mixing(config.hidden_size, residual_size)

    def __call__(self, x: Tensor, first_layer_value: typing.Optional[Tensor]) -> typing.Tuple[Tensor, typing.Optional[Tensor]]:
        '''
        Pass the tensor x through all of the modules in this layer.
        Following Llama 2's example, we use pre-normalization for improved gradient stability.
        '''

        attended_residual, first_layer_value = self.attn(self.h1_pre(self.ln_1(x)), first_layer_value)
        x = self.h1_res(x) + self.h1_post(attended_residual)

        x = self.h2_res(x) + self.mlp(self.h2_pre(self.ln_2(x)))

        return x, first_layer_value

class TransformerV2(nn.Module):
    def __init__(self, config: AilmV2Config):
        '''
        Initialize the Transformer block of the model.
        '''
        super().__init__()

        # Weights for Token Embeddings. Acts like a lookup table translating from token ids to embedding vectors.
        self.wte = nn.Embedding(config.vocab_size, config.residual_size)

        # All of the layers share one fixed copy of the Rotary Positional Embeddings.
        rope = nn.RoPE(config.hidden_size//config.head_count)
        rope.freeze()

        # The sequential layers in the model.
        if config.hyper_connection_residual_stream_expansion_factor == 1:
            self.h = nn.Sequential(*[LayerV2(config, rope) for _ in range(config.layer_count)])
        else:
            self.h = nn.Sequential(*[HyperConnection(config, LayerV2(config, rope)) for _ in range(config.layer_count)])

        # The final normalization for the transformer.
        self.ln_f = rms_norm

        self.wte.set_dtype(config.dtype)
        #self.ln_f.set_dtype(config.dtype)
        self.h.set_dtype(config.dtype)

class AilmV2(AilmV1):

    def __init__(self, config: AilmV2Config):
        '''
        Initialize the model.
        '''
        super().__init__(config)
        self.config = config

        # The bulk of our transformer model.
        self.transformer = TransformerV2(config)

        residual_size = config.residual_size

        # The language model head projection.
        self.lm_head = nn.Linear(residual_size, config.vocab_size, bias=False)
        self.lm_head.set_dtype(config.dtype)

        # Initialize the model's weights.
        self.apply_to_modules(_init_weights)
        self.set_dtype(config.dtype)

        # In order to reduce the number of parameters, we tie the input embeddings to the language model head.
        self.transformer.wte.weight = self.lm_head.weight


    def __call__(self, ids: Tensor) -> Tensor:
        '''
        Given a tensor of ids with shape (Batch, Sequence Length, hidden_size),
        pass them all the way through the model to result in an array of (Batch, Sequence Length, vocab_size) logits.
        Input ids should an integer type, probably uint16, and the resulting logits should have the type of config.dtype.
        '''

        # Lookup our embedding vectors corresponding to the input token ids.
        x: Tensor = self.transformer.wte(ids)

        first_layer_value: typing.Optional[Tensor] = None
        # Pass them through each layer of the transformer.
        for layer in self.transformer.h.layers:
            x, first_layer_value = layer(x, first_layer_value)

        # Normalize the last result before projecting them into logits.
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return logits

