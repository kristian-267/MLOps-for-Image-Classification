import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from model import ResNeStModel
from torch import optim
from torch.profiler import ProfilerActivity, profile
from pytorch_lightning import LightningModule


class LitDemucs(LightningModule):
    def __init__(self, wandb):
        super().__init__()
        self.config = wandb.config
        self.model = ResNeStModel()

        self.criterion = getattr(torch.nn, self.config.criterion)()
        self.scheduler = getattr(optim.lr_scheduler, self.config.scheduler)(
            self.optimizer, mode=self.config.lr_mode, factor=self.config.lr_decay, patience=self.config.lr_patience, threshold=self.config.lr_threshold
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_pred = self(x)
        y_pred = nn.LogSoftmax(dim=1)(y_pred)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        loss, accuracy = self._shared_eval_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_accuracy", accuracy, prog_bar=True, sync_dist=True)
        return loss, accuracy

    def test_step(self, batch):
        loss, accuracy = self._shared_eval_step(batch)
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_accuracy", loss, sync_dist=True)
        return loss, accuracy

    def _shared_eval_step(self, batch):
        x, y = batch
        y_pred = self(x)
        y_pred = nn.LogSoftmax(dim=1)(y_pred)
        loss = self.criterion(y_pred, y)

        ps = torch.exp(y_pred)
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == y.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))

        return loss, accuracy

    def configure_optimizers(self):
        optimizer = getattr(optim, self.config.optimizer)(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.decay,
            momentum=self.config.momentum,
        )
        optimizer_config = {
            "optimizer": optimizer,
        }
        optimizer_config["lr_scheduler"] = {
                "scheduler": ReduceLROnPlateau(optimizer),
                "monitor": self.args["reduce_on_plateau_metric"],
            }

        return optimizer_config
