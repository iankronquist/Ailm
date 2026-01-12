'''
Ailm - PyTorch Port
Model: Claude Sonnet 4.5 (2025-01-12)

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

Ported from MLX to PyTorch.
'''

import dataclasses
import math
from typing import Optional

import torch
from torch import nn
from torch import Tensor


DTYPE_NAME_TO_TORCH_DTYPE = {
    'bf16': torch.bfloat16,
    'bfloat16': torch.bfloat16,
    'f16': torch.float16,
    'float16': torch.float16,
    'f32': torch.float32,
    'float32': torch.float32,
}


def dtype_name_to_torch_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_NAME_TO_TORCH_DTYPE:
        raise ValueError(
            f"Unrecognized dtype '{name}'. "
            f"Valid options: {', '.join(DTYPE_NAME_TO_TORCH_DTYPE.keys())}"
        )
    return DTYPE_NAME_TO_TORCH_DTYPE[name]


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
    - dtype: The torch DType of the model. We recommend bfloat16 for performance or float32 for precision.
      Without additional gradient scaling, float16 will not have the necessary range to represent the gradients, and training will collapse into a pile of nans.
    - mlp_size: The inner dimension of the Multi-Layer Perceptron.
    - use_kv_cache: Whether to use Key Value Caching. We recommend keeping this off for training, and enabling it for inference.
    - use_attention_sinks: Whether to use attention sinks. Attention sinks allow the model to not pay attention to certain tokens. See [Attention sinks](https://arxiv.org/pdf/2309.17453).
    - no_rms_norm_weight: Disable weights on RMS norms. Empirically they are all set to 1. This saves a few parameters.
    '''

    use_kv_cache: bool = False
    vocab_size: int = 50304
    layer_count: int = 12
    head_count: int = 12
    hidden_size: int = 768
    mlp_size: int = 768 * 4
    key_value_head_count: int = 4
    dtype: torch.dtype = torch.bfloat16
    use_attention_sinks: bool = True
    no_rms_norm_weight: bool = True

    def to_dict(self):
        '''Convert self to a dictionary suitable for serializing to json or yaml'''
        dictionary = dataclasses.asdict(self)
        dictionary['dtype'] = str(dictionary['dtype'])
        return dictionary

    def __post_init__(self):
        # If dtype was provided as a name eg from a config file, translate it to the corresponding torch dtype.
        if isinstance(self.dtype, str):
            name = self.dtype.strip().lower()
            self.dtype = dtype_name_to_torch_dtype(name)


def _init_weights(module: nn.Module):
    '''
    Initialize the weights for the module.
    '''
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class Mlp(nn.Module):
    '''The Multi-Layer Perceptron.
    We use SiLU since it's more efficient than GELU.
    However, unlike Llama 2, we do not use a gating matrix in order to reduce our number of parameters.'''
    
    def __init__(self, config: AilmV1Config):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.mlp_size, bias=False)
        self.activation = nn.SiLU()
        self.c_proj = nn.Linear(config.mlp_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        return x


class RotaryPositionalEmbedding(nn.Module):
    '''Rotary Positional Embedding (RoPE) implementation for PyTorch'''

    inv_freq: Tensor
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
    def forward(self, q: Tensor, offset: int = 0) -> Tensor:
        '''Apply RoPE to query or key tensors'''
        seq_len = q.shape[2]
        
        # Generate position indices
        positions = torch.arange(offset, offset + seq_len, device=q.device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Reshape for broadcasting
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        
        # Apply rotation
        q_rot = self._rotate_half(q)
        q_embed = q * cos + q_rot * sin
        
        return q_embed
    
    def _rotate_half(self, x: Tensor) -> Tensor:
        '''Rotate half the hidden dims of the input'''
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)


def rms_norm(x: Tensor, eps: float = 1e-8) -> Tensor:
    '''Root Mean Square Layer Normalization'''
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class CausalSelfAttention(nn.Module):
    '''
    CausalSelfAttention
    
    We use Grouped Query Attention to save parameters: https://arxiv.org/pdf/2305.13245
    '''
    
    def __init__(self, config: AilmV1Config, rope: RotaryPositionalEmbedding):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_count = config.head_count
        self.key_value_head_count = config.key_value_head_count
        self.head_dim = config.hidden_size // config.head_count

        # Initialize the Wk, Wq, Wv linear projections packed together
        out_dim = self.hidden_size + 2 * self.head_dim * self.key_value_head_count
        self.c_attn = nn.Linear(self.hidden_size, out_dim, bias=False)

        # Initialize the output projection back to residual space
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Empty initial cache
        self.k_cache: Optional[Tensor] = None
        self.v_cache: Optional[Tensor] = None
        self.use_kv_cache = config.use_kv_cache

        # Hold onto the shared rope module
        self.rope = rope

        # Calculate our scale for our attention step
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Calculate the size of the Q, K, and V vectors
        self.q_size = self.head_count * self.head_dim
        self.kv_size = self.key_value_head_count * self.head_dim

        # Attention sinks
        if config.use_attention_sinks:
            self.register_buffer('sinks', torch.zeros(self.head_count, dtype=config.dtype))
        else:
            self.sinks = None

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape

        # Packed QKV projection
        qkv = self.c_attn(x)

        # Slice the queries, keys, and values
        q = qkv[:, :, :self.q_size]
        k = qkv[:, :, self.q_size:(self.q_size + self.kv_size)]
        v = qkv[:, :, (self.q_size + self.kv_size):]

        # Reshape for multi-head attention
        q = q.view(B, T, self.head_count, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.key_value_head_count, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.key_value_head_count, self.head_dim).transpose(1, 2)

        # Calculate offset for RoPE
        if self.use_kv_cache and self.k_cache is not None:
            offset = self.k_cache.shape[2]
        else:
            offset = 0

        # Apply Rotational Positional Embeddings
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # Update KV cache if enabled
        if self.use_kv_cache:
            if self.k_cache is None or self.v_cache is None:
                self.k_cache = k
                self.v_cache = v
            else:
                self.k_cache = torch.cat([self.k_cache, k], dim=2)
                self.v_cache = torch.cat([self.v_cache, v], dim=2)
            k = self.k_cache
            v = self.v_cache

        assert k is not None
        assert v is not None
        # Expand KV heads for grouped query attention
        if self.key_value_head_count < self.head_count:

            k = k.repeat_interleave(self.head_count // self.key_value_head_count, dim=1)
            v = v.repeat_interleave(self.head_count // self.key_value_head_count, dim=1)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if not self.use_kv_cache or T > 1:
            causal_mask = torch.triu(torch.ones(T, k.shape[2], device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask[None, None, :, :], float('-inf'))
        
        # Apply attention sinks if enabled
        if self.sinks is not None:
            # Add sink contribution (simplified implementation)
            attn = attn + self.sinks.view(1, -1, 1, 1)
        
        attn = torch.softmax(attn, dim=-1)
        y = attn @ v

        # Reshape back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Project back to residual stream
        y = self.c_proj(y)

        return y

    def reset_key_value_cache(self, enable_key_value_cache: bool):
        '''Reset the Key Value Cache for this attention module'''
        self.k_cache = None
        self.v_cache = None
        self.use_kv_cache = enable_key_value_cache


class Layer(nn.Module):
    '''A single layer in the model'''
    
    def __init__(self, config: AilmV1Config, rope: RotaryPositionalEmbedding):
        super().__init__()
        self.attn = CausalSelfAttention(config, rope)
        self.mlp = Mlp(config)

    def forward(self, x: Tensor) -> Tensor:
        '''Pass the tensor x through all modules in this layer'''
        x = x + self.attn(rms_norm(x))
        x = x + self.mlp(rms_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: AilmV1Config):
        '''Initialize the Transformer block of the model'''
        super().__init__()

        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)

        # Shared Rotary Positional Embeddings
        rope = RotaryPositionalEmbedding(config.hidden_size // config.head_count)
        
        # Sequential layers
        self.h = nn.ModuleList([Layer(config, rope) for _ in range(config.layer_count)])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.h:
            x = layer(x)
        return x


class AilmV1(nn.Module):
    def __init__(self, config: AilmV1Config):
        '''Initialize the AilmV1 model'''
        super().__init__()
        self.config = config

        # The bulk of our transformer model
        self.transformer = Transformer(config)

        # The language model head projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize the model's weights
        self.apply(_init_weights)

        # Tie input embeddings to the language model head
        self.lm_head.weight = self.transformer.wte.weight

        # Move to desired dtype
        self.to(config.dtype)

    def forward(self, ids: Tensor) -> Tensor:
        '''
        Given a tensor of ids with shape (Batch, Sequence Length),
        pass them through the model to result in logits of shape (Batch, Sequence Length, vocab_size).
        '''
        # Lookup embedding vectors
        x = self.transformer.wte(ids)

        # Pass through transformer layers
        x = self.transformer(x)

        # Final normalization
        x = rms_norm(x)

        # Project to vocabulary logits
        logits = self.lm_head(x)

        return logits

    def reset_key_value_cache(self, enable_key_value_cache: bool):
        '''Reset the Key Value Cache for all layers in this model'''
        for layer in self.transformer.h:
            layer.attn.reset_key_value_cache(enable_key_value_cache)