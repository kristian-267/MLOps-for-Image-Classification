import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from models.model import ResNeStModel

path = "data/processed"
visual_path = "reports/figures"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

batch_size = 64


@click.command()
@click.argument("model_checkpoint", type=click.File("rb"))
def visual_tsne(model_checkpoint):
    train_dataset = torch.load(path + "/train_dataset.pt")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )

    checkpoint = torch.load(model_checkpoint)

    model = ResNeStModel()
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    features = None
    lbs = []

    for images, labels in train_loader:
        imgs = images.to(device)
        lbs += labels

        with torch.no_grad():
            output = model.forward(imgs)

        current_features = output.cpu().numpy()

        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

    tsne = TSNE(
        n_components=2, learning_rate="auto", init="pca", verbose=2
    ).fit_transform(features)

    tx, ty = tsne[:, 0], tsne[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    labels = np.unique(np.array(lbs))
    num_labels = len(labels)
    print(num_labels)
    print(labels)
    for label in labels:
        indices = [i for i, l in enumerate(lbs) if l.item() == label]

        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        color = np.array(
            [
                [
                    label / (num_labels - 1),
                    1 - label / (num_labels - 1),
                    label / (num_labels - 1),
                ]
            ]
            * len(indices)
        )

        ax.scatter(current_tx, current_ty, c=color, label=label)

    plt.savefig(visual_path + "/tsne.png")


if __name__ == "__main__":
    # model_checkpoint = 'models/trained_model.pth'
    # pvisual_tsne(model_checkpoint)

    visual_tsne()
