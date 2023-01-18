from src.models.model import ResNeSt
import hydra
from hydra import compose
import torch


def main():
    hydra.initialize(config_path="../conf", job_name="predict")
    config = compose(config_name='predict.yaml')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNeSt.load_from_checkpoint("models/model.ckpt", map_location=device, hparams=config)
    script_model = model.to_torchscript(method='script')
    torch.jit.save(script_model, "model_store/deployable_model.pt")


if __name__ == "__main__":
    main()
