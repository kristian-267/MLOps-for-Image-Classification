from typing import Any, Dict, List, Tuple, Union

import omegaconf
import timm
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import optim


class ResNeSt(LightningModule):
    def __init__(self, hparams: omegaconf.DictConfig) -> None:
        super().__init__()
        self.params = hparams
        self.lr = self.params.lr
        self.model = timm.create_model(self.params.model, pretrained=False)
        self.model.apply(init_weights)
        self.criterion = getattr(torch.nn, self.params.criterion)()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.jit.ignore
    def training_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, Any], None]:
        x, y = batch
        y_pred = self(x)
        y_pred = self.logsoftmax(y_pred)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)

        return loss

    @torch.jit.ignore
    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, Any], None]:
        loss, accuracy = self._shared_eval_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, logger=True)
        self.log("val_accuracy", accuracy, prog_bar=True, sync_dist=True, logger=True)

        return {"val_loss": loss, "val_accuracy": accuracy}

    @torch.jit.ignore
    def test_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, Any], None]:
        loss, accuracy = self._shared_eval_step(batch)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True, logger=True)
        self.log("test_accuracy", accuracy, prog_bar=True, sync_dist=True, logger=True)

        return {"test_loss": loss, "test_accuracy": accuracy}

    @torch.jit.ignore
    def _shared_eval_step(
        self, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        y_pred = self(x)
        y_pred = self.logsoftmax(y_pred)
        loss = self.criterion(y_pred, y)

        ps = torch.exp(y_pred)
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == y.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.float))

        return loss, accuracy

    @torch.jit.ignore
    def configure_optimizers(self):
        optimizer = getattr(optim, self.params.optimizer)(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.decay,
            momentum=self.params.momentum,
        )
        optimizer_config = {
            "optimizer": optimizer,
            "lr_scheduler": getattr(optim.lr_scheduler, self.params.scheduler)(
                optimizer,
                mode=self.params.lr_mode,
                factor=self.params.lr_decay,
                patience=self.params.lr_patience,
                threshold=self.params.lr_threshold,
            ),
            "monitor": self.params.lr_monitor,
        }

        return optimizer_config


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

    if isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight_ih" in name:
                nn.init.kaiming_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
