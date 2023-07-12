try:
    from typing import Literal
except ModuleNotFoundError:
    from typing_extensions import Literal

import lightning
import torch

from vision.utils.load_weights import parse_state_dict


class BaseTask(lightning.LightningModule):
    def load_partial_state_dict(
        self, initial_weight_path: str, initial_weight_type: Literal["full", "backbone"]
    ):  # type: ignore
        org_state_dict = self.model.model.state_dict()
        trg_state_dict = torch.load(initial_weight_path)["state_dict"]
        trg_state_dict = parse_state_dict(trg_state_dict, initial_weight_type)  # type: ignore
        org_state_dict.update(trg_state_dict)
        self.model.model.load_state_dict(org_state_dict)
