# -*- coding: utf-8 -*-
import os
from pathlib import Path
import multiprocessing
import omegaconf
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional

CROPSIZE = 224
RESIZE = 256
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


class DataModule(pl.LightningDataModule):
    def __init__(self, config: omegaconf.DictConfig, parent: Optional['DataModule']=None, additional_transform=None) -> None:
        super().__init__()
        self.train_dir = os.path.join(
            config.paths.raw_data_path + config.data.name, "train"
        )
        self.val_dir = os.path.join(
            config.paths.raw_data_path + config.data.name, "val"
        )
        self.output_dir = config.paths.processed_data_path
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(CROPSIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(RESIZE),
                transforms.CenterCrop(CROPSIZE),
                transforms.ToTensor(),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
            ]
        )
        self.batch_size = config.experiment.batch_size
        self.threads = multiprocessing.cpu_count()
        self.parent = parent
        self.additional_transform = additional_transform


    def prepare_data(self) -> None:
        # load raw data and save
        if self.parent is None:
            train_dataset = datasets.ImageFolder(
                self.train_dir, self.train_transform
            )
            val_dataset = datasets.ImageFolder(self.val_dir, self.val_transform)

            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

            torch.save(train_dataset, self.output_dir + "train_dataset.pt")
            torch.save(val_dataset, self.output_dir + "val_dataset.pt")
        else:
            pass

    def setup(self, stage: str) -> datasets.ImageFolder:
        # Assign train/val datasets for use in dataloaders
        if self.parent is None:
            if stage == "fit":
                self.train = torch.load(self.output_dir + "train_dataset.pt")
                self.val = torch.load(self.output_dir + "val_dataset.pt")

                return self.train, self.val

            # Assign test dataset for use in dataloader(s)
            if stage == "test":
                self.test = torch.load(self.output_dir + "val_dataset.pt")

                return self.test

            if stage == "predict":
                self.predict = torch.load(self.output_dir + "val_dataset.pt")

                return self.predict
        else:
            if stage == "fit":
                self.train = self.parent.train
                self.val = self.parent.val

                return self.train, self.val

            # Assign test dataset for use in dataloader(s)
            if stage == "test":
                self.test = self.parent.test

                return self.test

            if stage == "predict":
                self.predict = self.parent.predict

                return self.predict
    
    def collate_fn(self, batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)
        if self.additional_transform:
            batch = (self.additional_transform(batch[0]), *batch[1:])
        return batch

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.threads, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.threads, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.threads, pin_memory=True, collate_fn=self.collate_fn)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict, batch_size=self.batch_size, shuffle=False, num_workers=self.threads, pin_memory=True, collate_fn=self.collate_fn
        )
    
    def default_dataloader(self, batch_size=None, num_samples=None, shuffle=True):
        dataset = self.val
        if batch_size is None:
            batch_size = self.batch_size
        replacement = num_samples is not None
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=replacement, num_samples=num_samples)
        else:
            sampler = None
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=self.collate_fn)
