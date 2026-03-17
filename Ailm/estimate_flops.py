from model import AilmV1Config

config = AilmV1Config()


def estimate_matmul(i: int, j: int, k: int) -> int:
    # ij,jk->ik
    return 2 * i * j * k
def estimate_attention(config: AilmV1Config, T: int) -> float:
    total_flops = 0.0
    head_size = config.hidden_size // config.head_count
    kv_size = config.key_value_head_count * head_size
    q_flops = estimate_matmul(T, config.hidden_size, config.hidden_size)
    k_flops = estimate_matmul(T, config.hidden_size, kv_size)
    v_flops = estimate_matmul(T, config.hidden_size, kv_size)

    attn_flops = estimate_matmul(T, config.hidden_size, T)

    av_flops = estimate_matmul(T, T, config.hidden_size)
    out_flops = estimate_matmul(T, config.hidden_size, config.hidden_size)

    return q_flops + k_flops + v_flops + attn_flops + av_flops + out_flops


def estimate_mlp(config: AilmV1Config, T: int) -> float:
    up_flops = estimate_matmul(config.mlp_size, config.hidden_size, T)
    gate_flops = estimate_matmul(config.mlp_size, T, config.mlp_size)
    down_flops = estimate_matmul(config.hidden_size, config.mlp_size, T)
    return up_flops + gate_flops + down_flops


def estimate_flops2(config: AilmV1Config, T: int):
    layer_flops = estimate_attention(config, T) + estimate_mlp(config, T)
    # lm_head_flops = estimate_matmul(T, config.hidden_size, config.vocab_size)
    lm_head_flops = 0
    return layer_flops * config.layer_count + lm_head_flops




def estimate_flops():
    """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
    sequence_len = 1024
    nparams = 114130944
    nparams_embedding = 0
    q = config.hidden_size // config.head_count
    num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * config.layer_count * config.head_count * q * sequence_len
    return num_flops_per_token
flops = (estimate_flops())
flops2 = (estimate_flops2(config, 1024))
print('flops', flops)
print('flops2', flops2)
print('flops diff pct', flops2/flops*100, '%')
tflops = flops/(1024**4)
print('tflops', tflops)
grad_accum_steps = 123
secs = (grad_accum_steps * 1024 * 4) * tflops
print('secs', secs)

lm_head = estimate_matmul(1024, config.hidden_size, config.vocab_size)
print('lm head', lm_head, lm_head > flops)