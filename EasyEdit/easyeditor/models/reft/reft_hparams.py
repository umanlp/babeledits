from dataclasses import dataclass
from typing import List
import yaml

from ...util.hparams import HyperParams


@dataclass
class ReFTHyperParams(HyperParams):
    # Model
    model_name: str
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    device: int
    alg_name: str
    vocab_type: str

    # Method
    layers: List[int]
    num_steps: int
    # batch_size: 1
    # max_length: 40
    lr: float
    component: str
    bf16: bool
    model_parallel: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config["alg_name"] == "BabelReFT") or print(
            f'BabelReFTHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} '
        )
        return cls(**config)
