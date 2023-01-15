import torch  # type: ignore
from hydra import compose, initialize  # type: ignore

from src.models.model import ResNeSt
from tests import IMAGENET_MINI_SHAPE, N_IMAGENET_MINI_CLASS


def test_model() -> None:
    with initialize(version_base=None, config_path="../conf"):
        config = compose(config_name="config.yaml")

    shape = IMAGENET_MINI_SHAPE[::-1]
    shape.append(config.experiment.batch_size)
    shape = shape[::-1]

    data = torch.rand(shape)
    model = ResNeSt(config.experiment)
    output = model.forward(data)

    assert list(output.shape) == [
        config.experiment.batch_size,
        N_IMAGENET_MINI_CLASS,
    ]


if __name__ == "__main__":
    test_model()
