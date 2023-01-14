from hydra import initialize, compose
from tests import N_IMAGENET_MINI_TRAIN, N_IMAGENET_MINI_VAL, IMAGENET_MINI_SHAPE, N_IMAGENET_MINI_CLASS
from src.data.make_dataset import DataModule


def test_data():
    with initialize(version_base=None, config_path="../conf"):
        config = compose(config_name="config.yaml")
    
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
