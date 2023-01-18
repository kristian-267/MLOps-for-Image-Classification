import glob
import json
import logging
import os

import click
import cv2
import hydra
import numpy as np
import torch
from hydra import compose

from src.data.make_dataset import CROPSIZE, IMGNET_MEAN, IMGNET_STD

IMAGE_EXT = [".png", "jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

log = logging.getLogger(__name__)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
def predict(input_filepath) -> None:
    hydra.initialize(config_path="../conf", job_name="predict")
    config = compose(config_name="predict.yaml")
    paths = config.paths

    files = glob.glob(input_filepath + "/*.*")
    images = []
    labels = []
    for file in files:
        filename, file_extension = os.path.splitext(file)
        if file_extension in IMAGE_EXT:
            img = cv2.imread(file)
            img = cv2.resize(img, (CROPSIZE, CROPSIZE))

            MEAN = 255 * np.array(IMGNET_MEAN)
            STD = 255 * np.array(IMGNET_STD)

            img = (img - MEAN) / STD
            img = img.T
            img = img[np.newaxis, :, :, :]
            img = torch.from_numpy(img).float()

            images.append(img.to(device))
            labels.append(filename)

        else:
            log("It's not an image file!")

    model = torch.jit.load(paths.model_path + "deployable_model.pt")
    model.to(device)
    model.eval()

    for i in range(len(images)):
        image = images[i]
        label = labels[i]

        output = model(image)
        ps = torch.exp(output)
        _, top_class = ps.topk(1, dim=1)

        outcome = mapping_to_outcome(top_class.item())

        print(f"The prediction result of image {label} is: {outcome}\n")


def mapping_to_outcome(top_class):
    with open("app/index_to_name.json") as f:
        data = json.load(f)
        f.close()

    outcome = data[str(top_class)][1]

    return outcome


if __name__ == "__main__":
    predict()
