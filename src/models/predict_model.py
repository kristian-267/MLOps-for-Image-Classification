from pathlib import Path

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelPruning, QuantizationAwareTraining
from torch.profiler import ProfilerActivity
from yaml.loader import SafeLoader

import wandb
from src.data.make_dataset import DataModule
from src.models.model import ResNeSt


@hydra.main(config_path="../../conf", config_name="predict.yaml")
def predict(config: omegaconf.DictConfig) -> None:
    paths = config.paths

    datamodule = DataModule(config)
    model = ResNeSt(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=paths.model_path + hparams.name,
        filename="{epoch:02d}-{val_accuracy:.4f}",
        monitor=hparams.monitor,
        mode=hparams.monitor_mode,
        every_n_epochs=hparams.check_every_n_epoch,
        save_on_train_epoch_end=False,
    )
    early_stopping_callback = EarlyStopping(
        monitor=hparams.monitor,
        patience=hparams.es_patience,
        verbose=True,
        mode=hparams.monitor_mode,
    )
    pruning = ModelPruning("l1_unstructured")
    quantization = QuantizationAwareTraining()

    '''
    profiler = PyTorchProfiler(
        dirpath=paths.profile_path + hparams.name,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        **{
            "schedule": torch.profiler.schedule(
                skip_first=0, wait=0, warmup=198, active=2, repeat=1
            ),
            "record_shapes": True,
            "profile_memory": True,
            "on_trace_ready": torch.profiler.tensorboard_trace_handler(
                paths.profile_path + hparams.name
            ),
        }
    )
    '''

    trainer = pl.Trainer(
        default_root_dir=paths.log_path + hparams.name,
        logger=wandb_logger,
        log_every_n_steps=hparams.log_freq,
        # profiler=profiler,
        devices=hparams.device,
        accelerator=hparams.accelerator,
        precision=hparams.precision,
        max_epochs=hparams.max_epochs,
        max_steps=hparams.max_steps,
        num_sanity_val_steps=hparams.num_sanity,
        val_check_interval=hparams.val_check_interval,
        callbacks=[checkpoint_callback, early_stopping_callback, pruning, quantization],
    )
    trainer.fit(model=model, datamodule=datamodule)


def main():
    predict()


if __name__ == "__main__":
    main()
