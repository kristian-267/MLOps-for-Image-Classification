import glob
import os

import click
import cv2
import numpy as np
import torch
import torch.nn as nn
from model import ResNeStModel
from data.make_dataset import IMGNET_MEAN, IMGNET_STD, CROPSIZE

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

IMAGE_EXT = [".png", "jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]


@click.command()
@click.argument("model_checkpoint", type=click.File("rb"))
@click.argument("input_filepath", type=click.Path(exists=True))
def predict(model_checkpoint, input_filepath):
    checkpoint = torch.load(model_checkpoint)
    model = ResNeStModel()
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    files = glob.glob(input_filepath + "/*.*")
    images = []
    labels = []
    for file in files:
        filename, file_extension = os.path.splitext(file)

        if file_extension in IMAGE_EXT:
            img = cv2.imread(file)
            img = cv2.normalize(
                cv2.resize(img, CROPSIZE), None, IMGNET_MEAN, IMGNET_STD, cv2.NORM_MINMAX
            ).T
            img = img[np.newaxis, :, :, :]

            images.append(torch.from_numpy(img.astype(np.float32)).to(device))
            labels.append(filename)

        elif file_extension == ".npz":
            data = np.load(file)
            images_metadata = data["images"]

            for i in range(images_metadata.shape[0]):
                img = images_metadata[i, :, :]
                img = cv2.normalize(
                    cv2.resize(img, CROPSIZE),
                    None,
                    IMGNET_MEAN,
                    IMGNET_STD,
                    cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U,
                )
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).T
                img = img[np.newaxis, :, :, :]

                images.append(torch.from_numpy(img.astype(np.float32)).to(device))
                labels.append(i)

    for i in range(len(images)):
        image = images[i]
        label = labels[i]

        output = nn.LogSoftmax(dim=1)(model(image))
        ps = torch.exp(output)
        _, top_class = ps.topk(1, dim=1)

        print(f"The prediction of image {label} is: {top_class.item()}")


if __name__ == "__main__":
    # model_checkpoint = 'models/trained_model.pth'
    # input_filepath = 'data/external'
    # predict(model_checkpoint, input_filepath)

    predict()
