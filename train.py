import random
import numpy as np
import torch
from vision.configs import get_config
from vision.trainer import get_trainer

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
    exp_config = get_config("dog_vs_cat_classification_resnet")
    trainer = get_trainer(exp_config)
    trainer.train_and_eval()
