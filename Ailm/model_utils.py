import typing
import mlx.core

from model import AilmV1, AilmV1Config
from modelv2 import AilmV2, AilmV2Config
from model_attention_residuals import AilmAttentionResidualsConfig, AilmAttentionResiduals

def model_name_to_class_and_config(name: str) -> typing.Tuple[type[AilmV1], type[AilmV1Config]]:
    if name == 'AilmV1':
        return AilmV1, AilmV1Config
    elif name == 'AilmV2':
        return AilmV2, AilmV2Config
    elif name == 'AilmAttentionResiduals':
        return AilmAttentionResiduals, AilmAttentionResidualsConfig
    raise NotImplementedError(f"Unimplemented model named {name}")
