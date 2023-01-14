import hydra
import torch
from tests import IMAGENET_MINI_SHAPE, N_IMAGENET_MINI_CLASS
from src.models.model import ResNeSt


@hydra.main(config_path="../conf", config_name="config.yaml")
def test_model(config):
    shape = IMAGENET_MINI_SHAPE[::-1]
    shape.append(config.experiment.batch_size)
    shape = shape[::-1]

    data = torch.rand(shape)
    model = ResNeSt(config.experiment)
    output = model.forward(data)

    assert list(output.shape) == [config.experiment.batch_size, N_IMAGENET_MINI_CLASS]


if __name__ == "__main__":
    test_model()
