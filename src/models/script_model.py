from src.models.model import ResNeSt
import hydra
from hydra import compose
import torch


def main():
    hydra.initialize(config_path="../../conf", job_name="predict")
    config = compose(config_name='predict.yaml')

    model = ResNeSt.load_from_checkpoint(config.paths.model_path + "model.ckpt", map_location=torch.device("cpu"), hparams=config)
    script_model = model.to_torchscript(method='script')
    torch.jit.save(script_model, config.paths.model_path + "deployable_model.pt")


if __name__ == "__main__":
    main()
