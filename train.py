import argparse
import os.path
import random

import numpy as np
import torch
from vision.configs import get_experiment_config
from vision.serving.export import ModelExporter, ExportType
from vision.trainer.trainer import BaseTrainer
import project  # noqa: F401

seed = 2021
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    config = argparse.ArgumentParser()

    config.add_argument("--exp_name", type=str)
    config.add_argument("--log_dir", type=str)
    config.add_argument("--label", type=str)

    args = config.parse_args()

    exp_config = get_experiment_config(args.exp_name)

    trainer = BaseTrainer(exp_config)
    trainer.train_and_eval()

    exporter = ModelExporter(
        exp_config,
        checkpoint_path=os.path.join(".", exp_config.log_dir, "best.ckpt"),
        export_dir=os.path.join(".", "export", exp_config.log_dir),
    )
    exporter.export(ExportType.onnx)
    exporter.export(ExportType.jit)
