import hydra
from tests import N_IMAGENET_MINI_TRAIN, N_IMAGENET_MINI_VAL, IMAGENET_MINI_SHAPE, N_IMAGENET_MINI_CLASS
from src.data.make_dataset import DataModule


@hydra.main(config_path="../conf", config_name="config.yaml")
def test_data(config):
    datamodule = DataModule(config)
    datamodule.prepare_data()
    trainset, valset = datamodule.setup('fit')

    assert len(trainset) == N_IMAGENET_MINI_TRAIN
    assert len(valset) == N_IMAGENET_MINI_VAL

    train_labels = {}
    val_labels = {}

    for x, y in iter(trainset):
        assert list(x.shape) == IMAGENET_MINI_SHAPE

        if y not in train_labels.keys():
            train_labels.update({y: True})
    
    for x, y in iter(valset):
        assert list(x.shape) == IMAGENET_MINI_SHAPE

        if y not in val_labels.keys():
            val_labels.update({y: True})
    
    assert len(train_labels.keys()) == N_IMAGENET_MINI_CLASS
    assert len(val_labels.keys()) == N_IMAGENET_MINI_CLASS


if __name__ == "__main__":
    test_data()
