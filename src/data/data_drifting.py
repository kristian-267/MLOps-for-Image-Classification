# -*- coding: utf-8 -*-
import copy
import logging
from collections import OrderedDict

import hydra
import matplotlib.pyplot as plt
import sklearn.manifold
import timm
import torch
import torchdrift
from hydra import compose

from src.data.make_dataset import DataModule
from src.models.model import ResNeSt

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

log = logging.getLogger(__name__)


def fit_detector(model, dataloader):
    detector = torchdrift.detectors.KernelMMDDriftDetector(return_p_value=True)
    feature_extractor = copy.deepcopy(model)
    feature_extractor[1] = torch.nn.Identity()
    torchdrift.utils.fit(
        dataloader, feature_extractor, detector, num_batches=1
    )
    return detector, feature_extractor


def main():
    hydra.initialize(config_path="../../conf", job_name="config")
    config = compose(config_name="config.yaml")

    datamodule = DataModule(config)
    datamodule.threads = 1
    datamodule.batch_size = 32
    datamodule.setup("fit")

    model = ResNeSt.load_from_checkpoint(
        config.paths.model_path + "model.ckpt",
        map_location=device,
        hparams=config.experiment,
    )
    state_dict = OrderedDict(
        [(k[6:], v) for k, v in model.state_dict().items()]
    )

    feature_extractor = timm.create_model(
        config.experiment.model, pretrained=False
    )
    feature_extractor.load_state_dict(state_dict)
    feature_extractor.fc = torch.nn.Identity()
    model = torch.nn.Sequential(
        feature_extractor,
        torch.nn.Linear(2048, 2),
    )
    model.eval().to(device)

    for p in model.parameters():
        p.requires_grad_(False)

    detector, feature_extractor = fit_detector(
        model, datamodule.train_dataloader()
    )

    batch = next(iter(datamodule.val_dataloader()))[0].to(device)
    batch_drifted = torchdrift.data.functional.gaussian_blur(batch, 2)

    features = feature_extractor(batch)
    score = detector(features)
    p_val = detector.compute_p_value(features)

    mapper = sklearn.manifold.Isomap(n_components=2)
    base_embedded = mapper.fit_transform(detector.base_outputs)
    features_embedded = mapper.transform(features)

    plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c="r")
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    plt.title(f"score {score:.2f} p-value {p_val:.2f}")
    plt.savefig("reports/figures/drift_origin.png")

    features = feature_extractor(batch_drifted)
    score = detector(features)
    p_val = detector.compute_p_value(features)

    features_embedded = mapper.transform(features)

    plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c="r")
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    plt.title(f"score {score:.2f} p-value {p_val:.2f}")
    plt.savefig("reports/figures/drift_drifted.png")


if __name__ == "__main__":
    main()
