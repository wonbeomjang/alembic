from collections import OrderedDict
from typing import Dict

import torch
from torch import Tensor, nn

try:
    from typing import Literal
except ModuleNotFoundError:
    from typing_extensions import Literal


def load_partial_state_dict(
    self,
    model: nn.Module,
    initial_weight_path: str,
    initial_weight_type: Literal["full", "backbone"],
):  # type: ignore
    org_state_dict = model.model.state_dict()
    trg_state_dict = torch.load(initial_weight_path)["state_dict"]
    trg_state_dict = parse_state_dict(trg_state_dict, initial_weight_type)  # type: ignore
    org_state_dict.update(trg_state_dict)
    model.load_state_dict(org_state_dict)


def parse_state_dict(state_dict: Dict[str, Tensor], param_type: Literal["full", "backbone"]):  # type: ignore
    if param_type == "full":
        param_type = ""

    new_state_dict = OrderedDict()
    for name, param in state_dict.items():
        if param_type not in name:
            continue
        new_state_dict[name.replace("model.", "")] = param
    return new_state_dict
