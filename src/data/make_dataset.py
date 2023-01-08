# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import torch
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    traindir = os.path.join(input_filepath, "train")
    valdir = os.path.join(input_filepath, "val")
    normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    torch.save(train_dataset, output_filepath + "/train_dataset.pt")
    torch.save(val_dataset, output_filepath + "/val_dataset.pt")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # input_filepath = 'data/raw/imagenet-mini'
    # output_filepath = 'data/processed'
    # main(input_filepath, output_filepath)

    main()
