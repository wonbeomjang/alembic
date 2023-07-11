import os.path
import argparse
import random

import numpy as np
import torch
from vision.configs import get_config
from vision.trainer import get_trainer
from vision.configs import experiment

seed = 2021
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    config = argparse.ArgumentParser()

    config.add_argument("--initial_weight_path", type=str, default=None)
    config.add_argument("--initial_weight_type", type=str, default="backbone")
    config.add_argument("--log_dir", type=str)
    config.add_argument("--label", type=str)

    args = config.parse_args()

    exp_config = get_config("dagm_classification_resnet")

    exp_config.initial_weight_path = args.initial_weight_path
    exp_config.initial_weight_type = args.initial_weight_type

    exp_config.log_dir = args.log_dir
    label = args.label

    exp_config.train_data.label_path = os.path.join(
        experiment.DATASET_BASE_DIR, experiment.DAGM_DIR, label
    )

    exp_config.val_data.label_path = os.path.join(
        experiment.DATASET_BASE_DIR, experiment.DAGM_DIR, label
    )

    trainer = get_trainer(exp_config)
    trainer.train_and_eval()
