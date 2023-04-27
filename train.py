from vision.configs import get_config
from vision.trainer import get_trainer


if __name__ == "__main__":
    exp_config = get_config("neurocle_cla_led")
    trainer = get_trainer(exp_config)
    trainer.train_and_eval()
