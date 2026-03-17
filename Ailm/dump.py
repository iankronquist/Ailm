    #accumulation_type: typing.Optional[str]

def dtype_name_to_mlx(name: str) -> mx.Dtype:
    '''Translate a string name from a config file to the corresponding MLX DType.'''

    table = {
        'bf16': mx.bfloat16,
        'f16': mx.float16,
        'f32': mx.float32,
    }
    dtype = table.get(name)
    if not dtype:
        raise NameError(f"Unknown dtype name {name}, expected one of {', '.join(table.keys())}")
    return dtype