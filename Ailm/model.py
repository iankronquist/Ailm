'''
Ailm is a small Large Language Model designed to be easily trained on an M1 Max Macbook Pro.

It is named after the letter ailm from the Ogham alphabet which was used to write Old Irish:
https://en.wikipedia.org/wiki/Ailm

I drew upon the following resources to design this model:
    - Andrej Karpathy's nanoGPT
    - Andrej Karpathy's nanoChat
    - HuggingFace's SmolLM
    - Facebook's Llama 2
    - GPT-2
    - Pleia's Baguettotron

It uses Apple's MLX framework as it seems to get better performance than other frameworks.

'''

import dataclasses
import math
from abc import ABC, abstractmethod

import mlx
from mlx import nn
from mlx.core import array as Tensor
import mlx.core


@dataclasses.dataclass
class AilmV1Config:
    '''
    Configuration parameters for the model.
    These control size and types of the model weights, and not the optimizer or training hyperparameters.

    Members:
    - vocab_size: The dimension of the lm_head and embedding matrices. Must be big enough to accept the full range of the tokens.
      For example If you're using a tokenizer like GPT2 with a bit over 50k tokens, you'll need more than 50k here.
      In many cases we get better performance by rounding up to a power of two. This also gives space for a few extra thinking tokens if you need them.
    - layer_count: The number of layers in the model. I've read that smaller models tend to benefit from deeper layers.
    - head_count: The number of query heads in the model. It must evenly divide n_embed.
    - hidden_size: The dimension of the embeddings and hidden activations of the model.
    - key_value_head_count: The number of Key and Value heads. To save parameters, we use [Group Query Attention](https://arxiv.org/pdf/2305.13245).
    - dtype: The mlx DType of the model. We recommend bfloat16 for performance or float32 for precision.
      Without additional gradient scaling, float16 will not have the necessary range to represent the gradients, and training will collapse into a pile of nans.
    - mlp_size: The inner dimension of the Multi-Layer Perceptron.
    - use_kv_cache: Whether to use Key Value Caching. We recommend keeping this off for training, and enabling it for inference.
    - use_attention_sinks: Whether to use attention sinks. Attention sinks allow the model to not pay attention to certain tokens. See [Attention sinks](https://arxiv.org/pdf/2309.17453).
    - no_rms_norm_weight: Disable weights on RMS norms. Empirically they are all set to 1. This saves a few parameters.
    '''

    # This config is more like baguettotron. It's leaner and deeper.
    # At our current rate of 5.5k tokens per second, with this config it will take us just over 3 days
    # vocab_size: int = 50304
    # layer_count: int = 30
    # head_count: int = 9
    # hidden_size: int = 576
    # key_value_head_count: int = 3
    # dtype: mlx.core.Dtype = mlx.core.bfloat16
    # mlp_size: int = 1536
    # use_kv_cache: bool = False
    # use_attention_sinks: bool = True

    # This config is more like gpt2 small.
    # At our current rate of 7.5k tokens per second, with this config it will take us just over 3 days
    use_kv_cache: bool = False
    vocab_size: int = 50304
    layer_count: int = 12
    head_count: int = 12
    hidden_size: int = 768
    mlp_size: int = 768*4
    key_value_head_count: int = 4
    dtype: mlx.core.Dtype = mlx.core.bfloat16
    use_attention_sinks: bool = True
    no_rms_norm_weight: bool = True

    # This config is more like gpt2 medium.
    # At our current rate of 2.4k tokens per second with this config it will take us 28 days on my M1 max
    # use_kv_cache: bool = False
    # vocab_size: int = 50304
    # layer_count: int = 24
    # head_count: int = 16
    # hidden_size: int = 1024
    # mlp_size: int = 1024*4
    # key_value_head_count: int = 8
    # dtype: mlx.core.Dtype = mlx.core.bfloat16
    # use_attention_sinks: bool = True

    def to_dict(self):
        '''Convert self to a dictionary suitable for serializing to json or yaml'''
        dictionary = dataclasses.asdict(self)
        dictionary['dtype'] = str(dictionary['dtype'])
        return dictionary


def _init_weights(_name: str, module: nn.Module):
    '''
    Initialize the weights for the module.
    '''
    if isinstance(module, nn.Linear):
        module.weight[:] = mlx.core.random.normal(
            shape=module.weight.shape,
            loc=0.0,
            scale=0.02
        )
        if module.get('bias') is not None:
            module.bias[:] = mlx.core.zeros_like(module.bias)
    elif isinstance(module, nn.Embedding):
        module.weight[:] = mlx.core.random.normal(
            shape=module.weight.shape,
            loc=0.0,
            scale=0.02
        )
    # We don't need to initialize RMSNorm because they're initialized with all 1s


class Mlp(nn.Module):
    '''The Multi-Layer Perceptron.
    #We follow GPT2's design and use the GELU activation function.
    We use SilU since it's more efficient than GELU.
    However, unlike Llama 2, we do not use a gating matrix in order to reduce our number of parameters.'''
    def __init__(self, config: AilmV1Config):
        super().__init__()

        self.c_fc = nn.Linear(config.hidden_size, config.mlp_size, bias=False)
        self.activation = nn.SiLU()
        self.c_proj = nn.Linear(config.mlp_size, config.hidden_size, bias=False)
 
        self.c_fc.set_dtype(config.dtype)
        self.activation.set_dtype(config.dtype)
        self.c_proj.set_dtype(config.dtype)
    def __call__(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        return x



class CausalSelfAttention(nn.Module):
    '''
    CausalSelfAttention
    
    We use Grouped Query Attention to save parameters: https://arxiv.org/pdf/2305.13245
    '''
    def __init__(self, config: AilmV1Config, rope: nn.RoPE):
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


    def __call__(self, x: Tensor) -> Tensor:
        # We expect this hidden tensor x to have the dimensions for the Batch length, sequence length (T), and Channels (hidden_size)
        B, T, C = x.shape

        # The Wq, Wk, an Wv weights are all packed together so we can dispatch on big matrix multiply to the GPU instead of three smaller ones.
        qkv = self.c_attn(x)

        # Slice the queries, keys, and values out of the packed result.
        q = qkv[:, :, :self.q_size]
        k = qkv[:, :, self.q_size:(self.q_size+self.kv_size)]
        v = qkv[:, :, (self.q_size+self.kv_size):]

        # Reshape them to the layout we expect for rope and group query attention.
        q = q.reshape(B, T, self.head_count,  self.head_dim)
        q = mlx.core.swapaxes(q, 1, 2)
        k = k.reshape(B, T, self.key_value_head_count, self.head_dim)
        k = mlx.core.swapaxes(k, 1, 2)
        v = v.reshape(B, T, self.key_value_head_count, self.head_dim)
        v = mlx.core.swapaxes(v, 1, 2)

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

        # Apply Rotational Positional Embeddings.
        q = self.rope(q)
        k = self.rope(k)

        # Scaled dot product attention. Apply causal masking if we're not using KV caching.
        mask = None if self.use_kv_cache else "causal"
        y = mlx.core.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask, sinks=self.sinks)

        # Performance optimization: If we're using KV caching, we don't actually need to send any of the cached context on from here.
        if self.use_kv_cache:
            y = y[:, :, -1:, :].transpose(0, 2, 1, 3).reshape([B, 1, C])
        else:
            # Reshape the tensor so it's laid out like the original residual stream.
            y = y.transpose(0, 2, 1, 3).reshape([B, T, C])

        # Project back to the residual stream space.
        y = self.c_proj(y)

        return y

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

class Layer(nn.Module):
    '''
    A single layer in the model.
    '''
    def __init__(self, config: AilmV1Config, rope: nn.RoPE):
        '''
        Initialize a layer. The argument rope must be a nn.RoPE module.
        '''
        super().__init__()

        self.attn = CausalSelfAttention(config, rope)
        #self.ln_1 = nn.RMSNorm(config.hidden_size)
        self.ln_1 = rms_norm
        self.mlp = Mlp(config)
        #self.ln_2 = nn.RMSNorm(config.hidden_size)
        self.ln_2 = rms_norm
        #if config.no_rms_norm_weight:
        #    # The inferred type is incorrect, it should be an Optional[mlx.array]
        #    self.ln_1.weight = None # pyright: ignore reportAttributeAccessIssue
        #    self.ln_2.weight = None # pyright: ignore reportAttributeAccessIssue

    def __call__(self, x: Tensor) -> Tensor:
        '''
        Pass the tensor x through all of the modules in this layer.
        Following Llama 2's example, we use pre-normalization for improved gradient stability.
        '''

        x = x + self.attn(self.ln_1(x))

        x = x + self.mlp(self.ln_2(x))

        return x



class Transformer(nn.Module):
    def __init__(self, config: AilmV1Config):
        '''
        Initialize the Transformer block of the model.
        '''
        super().__init__()

        # Weights for Token Embeddings. Acts like a lookup table translating from token ids to embedding vectors.
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)

        # All of the layers share one fixed copy of the Rotary Positional Embeddings.
        rope = nn.RoPE(config.hidden_size//config.head_count)
        rope.freeze()

        # The sequential layers in the model.
        self.h = nn.Sequential(*[Layer(config, rope) for _ in range(config.layer_count)])

        # The final normalization for the transformer.
        self.ln_f = rms_norm
        #self.ln_f = nn.RMSNorm(config.hidden_size)
        #if config.no_rms_norm_weight:
        #    # The inferred type is incorrect, it should be an Optional[mlx.array]
        #    self.ln_f.weight = None # pyright: ignore reportAttributeAccessIssue

        self.wte.set_dtype(config.dtype)
        #self.ln_f.set_dtype(config.dtype)
        self.h.set_dtype(config.dtype)

class AilmV1(nn.Module):

    def __init__(self, config: AilmV1Config):
        '''
        Initialize the AilmV1 model.
        '''
        super().__init__()
        self.config = config

        # The bulk of our transformer model.
        self.transformer = Transformer(config)

        # The language model head projection.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.set_dtype(config.dtype)

        # Initialize the model's weights.
        self.apply_to_modules(_init_weights)

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

        # Pass them through each layer of the transformer.
        x = self.transformer.h(x)

        # Normalize the last result before projecting them into logits.
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return logits

    def reset_key_value_cache(self, enable_key_value_cache: bool):
        '''
        Reset the Key Value Cache for all layers in this model.

        If enable_key_value_cache is True, future calls to this module will cache previous inference values, which is recommended during inference, but not during training.
        Otherwise, if enable_key_value_cache is False, no caching will occur.
        '''
        for layer in self.transformer.h.layers:
            layer.attn.reset_key_value_cache(enable_key_value_cache)

