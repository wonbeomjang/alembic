import argparse
import random

import numpy as np
import torch
from vision.configs import get_experiment_config
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

    config.add_argument("--initial_weight_path", type=str, default=None)
    config.add_argument("--initial_weight_type", type=str, default="backbone")
    config.add_argument("--log_dir", type=str)
    config.add_argument("--label", type=str)

    args = config.parse_args()

    exp_config = get_experiment_config("dog_vs_cat_classification_resnet")
    trainer = BaseTrainer(exp_config)

    trainer.train_and_eval()
