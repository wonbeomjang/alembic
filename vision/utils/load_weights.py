from collections import OrderedDict
from typing import Dict

from torch import Tensor

try:
    from typing import Literal
except ModuleNotFoundError:
    from typing_extensions import Literal


def parse_state_dict(state_dict: Dict[str, Tensor], param_type: Literal["full", "backbone"]):  # type: ignore
    if param_type == "full":
        param_type = ""

    new_state_dict = OrderedDict()
    for name, param in state_dict.items():
        if param_type not in name:
            continue
        new_state_dict[name.replace("model.", "")] = param
    return new_state_dict
