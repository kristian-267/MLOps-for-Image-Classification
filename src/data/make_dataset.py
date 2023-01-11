# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import torch
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms
import hydra


CROPSIZE = 224
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


@hydra.main(config_path="../../conf", config_name='config.yaml')
def main(config):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    paths = config.paths
    data = config.data

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    traindir = os.path.join(paths.raw_data_path + data.name, "train")
    valdir = os.path.join(paths.raw_data_path + data.name, "val")
    normalize = transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD)

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([transforms.RandomResizedCrop(CROPSIZE), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(CROPSIZE),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    torch.save(train_dataset, paths.processed_data_path + "/train_dataset.pt")
    torch.save(val_dataset, paths.processed_data_path + "/val_dataset.pt")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
