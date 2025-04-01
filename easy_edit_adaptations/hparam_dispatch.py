import sys
sys.path.append('EasyEdit')

from EasyEdit.easyeditor.models import (
    ROMEHyperParams,
    MEMITHyperParams,
    IKEHyperParams,
    MENDHyperParams,
    FTHyperParams,
    R_ROMEHyperParams,
    BabelReFTHyperParams,
    GraceHyperParams,
    ReFTHyperParams
)
from EasyEdit.easyeditor.util import HyperParams

methods = {
    "rome": ROMEHyperParams,
    "r-rome": R_ROMEHyperParams,
    "memit": MEMITHyperParams,
    "ike": IKEHyperParams,
    "mend": MENDHyperParams,
    "ft": FTHyperParams,
    "babelreft": BabelReFTHyperParams,
    "reft": ReFTHyperParams,
    "grace": GraceHyperParams,
}

def get_hparm_class(method: str) -> type[HyperParams]:
    method = method.lower()

    if method not in methods:
        raise NotImplemented(f"method {method} is not among implemented methods: {list(methods.keys())}")

    return methods[method]


