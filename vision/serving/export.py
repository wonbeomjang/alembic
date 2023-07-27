import enum
import os
from typing import Optional

import torch.jit
from torch import nn

from utils.logger import console_logger
from vision.configs.experiment import ExperimentConfig
from vision.task.api import get_task
from vision.task.base import BaseTask


class ExportType(enum.Enum):
    jit = enum.auto()
    onnx = enum.auto()


class ExportModule(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class ModelExporter:
    def __init__(
        self,
        exp_cfg: ExperimentConfig,
        checkpoint_path: str,
        export_dir: str = "./export",
        export_module: Optional[ExportModule] = None,
    ):
        task: BaseTask = get_task(exp_cfg.task)
        task.eval()

        if checkpoint_path:
            task.load_from_checkpoint(
                checkpoint_path, config=getattr(exp_cfg.task, exp_cfg.task.type)
            )

        self.model: nn.Module = task.model

        if hasattr(self.model, "backbone") and hasattr(
            self.model.backbone, "reparameterize"
        ):
            console_logger.info("reparameterize backbone")
            self.model.backbone.reparameterize()

        if export_module is not None:
            self.model = export_module(self.model)

        self.input_shape = exp_cfg.val_data.image_size
        self.export_dir = export_dir

        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

    def export(self, export_type):
        if export_type == ExportType.jit:
            jit_script_model = torch.jit.script(self.model)
            jit_script_model.save(os.path.join(self.export_dir, "jit_export.pt"))
        elif export_type == ExportType.onnx:
            torch.onnx.export(
                self.model,
                torch.rand((1, *self.input_shape)),
                os.path.join(self.export_dir, "onnx_export.onnx"),
                export_params=True,
                opset_version=10,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
        else:
            raise ValueError(f"Export type error; get {export_type}")
