# -*- coding: utf-8 -*-
import torch
import hydra
from hydra import compose
import torchdrift
from src.data.make_dataset import DataModule
from src.models.model import ResNeSt
from src.models.data_drift_model import Classifier
import copy
import timm
from collections import OrderedDict
from matplotlib import pyplot
import sklearn


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ModelMonitor:
    def __init__(self, drift_detector, feature_layer, N = 20, callback = None, callback_interval = 1):
        self.N = N
        base_outputs = drift_detector.base_outputs
        self.drift_detector = drift_detector
        assert base_outputs is not None, "fit drift detector first"
        feature_dim = base_outputs.size(1)
        self.feature_rb = torch.zeros(N, feature_dim, device=base_outputs.device, dtype=base_outputs.dtype)
        self.have_full_round = False
        self.next_idx = 0
        self.hook = feature_layer.register_forward_hook(self.collect_hook)
        self.counter = 0
        self.callback = callback
        self.callback_interval = callback_interval

    def unhook(self):
        self.hook.remove()

    def collect_hook(self, module, input, output):
        self.counter += 1
        bs = output.size(0)
        if bs > self.N:
            output = output[-self.N:]
            bs = self.N
        output = output.reshape(bs, -1)
        first_part = min(self.N - self.next_idx, bs)
        self.feature_rb[self.next_idx: self.next_idx + first_part] = output[:first_part]
        if first_part < bs:
            self.feature_rb[: bs - first_part] = self.output[first_part:]
        if not self.have_full_round and self.next_idx + bs >= self.N:
            self.have_full_round = True
        self.next_idx = (self.next_idx + bs) % self.N
        if self.callback and self.have_full_round and self.counter % self.callback_interval == 0:
            p_val = self.drift_detector(self.feature_rb)
            self.callback(p_val)

    def plot(self):
        import sklearn.manifold
        from matplotlib import pyplot

        mapping = sklearn.manifold.Isomap()
        ref = mapping.fit_transform(self.drift_detector.base_outputs.to("cpu").numpy())

        test = mapping.transform(self.feature_rb.to("cpu").numpy())
        pyplot.scatter(ref[:, 0], ref[:, 1])
        pyplot.scatter(test[:, 0], test[:, 1])


def corruption_function(x: torch.Tensor):
    return torchdrift.data.functional.gaussian_blur(x, severity=2)

def fit_detector(model, dataloader):
    detector = torchdrift.detectors.KernelMMDDriftDetector(return_p_value=True)
    feature_extractor = model[:-1]  # without the fc layer
    torchdrift.utils.fit(dataloader, feature_extractor, detector, num_batches=1)
    return detector

def alarm(p_value):
    assert p_value > 0.01, f"Drift alarm! p-value: {p_value*100:.03f}%"

def main():
    hydra.initialize(config_path="../../conf", job_name="config")
    config = compose(config_name='config.yaml')

    stage = 'fit'

    datamodule = DataModule(config)
    datamodule.setup(stage)
    ind_datamodule = datamodule
    ood_datamodule = DataModule(config, parent=datamodule, additional_transform=corruption_function)
    ood_datamodule.setup(stage)

    model = ResNeSt.load_from_checkpoint(config.paths.model_path + "model.ckpt", map_location=device, hparams=config.experiment)
    state_dict = OrderedDict([(k[6:], v) for k, v in model.state_dict().items()])

    feature_extractor = timm.create_model(config.experiment.model, pretrained=False)
    feature_extractor.load_state_dict(state_dict)
    model = torch.nn.Sequential(
        feature_extractor,
        feature_extractor.fc
    )
    feature_extractor.fc = torch.nn.Identity()
    model.eval().to(device)

    for p in model.parameters():
        p.requires_grad_(False)
    
    detector = fit_detector(model, datamodule.train_dataloader())

    it = iter(ind_datamodule.val_dataloader())
    batch = next(it)[0].to(device)
    batch_drifted = torchdrift.data.functional.gaussian_blur(next(it)[0].to(device), 5)

    res = model(batch).argmax(1)
    
    detector.compute_p_value(mm.feature_rb)
    
    feature_extractor.fc = torch.nn.Identity()

    model = Classifier(feature_extractor)
    model.to(device)

    feature_extractor = copy.deepcopy(model)
    feature_extractor.classifier = torch.nn.Identity()
    feature_extractor.to(device)

    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
    drift_detector.to(device)

    inputs, _ = next(iter(datamodule.train_dataloader()))

    torchdrift.utils.fit(datamodule.val_dataloader(), feature_extractor, drift_detector)

    drift_detection_model = torch.nn.Sequential(
        feature_extractor,
        drift_detector
    )

    features = feature_extractor(inputs)
    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)

    N_base = drift_detector.base_outputs.size(0)
    mapper = sklearn.manifold.Isomap(n_components=2)
    base_embedded = mapper.fit_transform(drift_detector.base_outputs)
    features_embedded = mapper.transform(features)
    pyplot.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
    pyplot.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    pyplot.title(f'score {score:.2f} p-value {p_val:.2f}');

    features = feature_extractor(inputs_ood)
    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)

    features_embedded = mapper.transform(features)
    pyplot.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
    pyplot.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    pyplot.title(f'score {score:.2f} p-value {p_val:.2f}');

    print("end")


if __name__ == "__main__":
    main()