import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from model import MyAwesomeModel

path = "data/processed"
model_checkpoint = "models/trained_model.pth"
visual_path = "reports/figures"

batch_size = 64
epoch = 1
eval_every = 100
stop_after = 30

decay = 1e-5
lr = 1e-3

criterion = nn.NLLLoss()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def train():
    train_dataset = torch.load(path + "/train_dataset.pt")
    val_dataset = torch.load(path + "/val_dataset.pt")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    _, _, H, W = next(iter(train_loader))[0].shape

    model = MyAwesomeModel(H, W)
    model.to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    train_losses, eval_losses, accuracies, steps = [], [], [], []
    train_loss = 0
    step = 0
    count = 0

    model.train()
    for e in range(epoch):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_losses.append(train_loss / (step + 1))

            if step % eval_every == 0:
                steps.append(step)

                eval_loss, accuracy = evaluate(model, val_loader)

                eval_losses.append(eval_loss)
                accuracies.append(accuracy)

                print(
                    f"Epoch: {e}/{epoch}\tStep: {step}\tTrain Loss: {train_losses[-1]}\tEval Loss: {eval_losses[-1]}\tAccuracy: {accuracies[-1]}%"
                )

                if len(accuracies) > 1 and accuracies[-1] <= accuracies[-2]:
                    count += 1
                else:
                    count = 0

                if count >= stop_after:
                    print("It's time for early stopping. Let's save the model!")
                    torch.save(
                        {"model_state_dict": model.state_dict(), "H": H, "W": W},
                        model_checkpoint,
                    )
                    break

            step += 1

        else:
            continue

        break
    else:
        print("Finish training and save the model.")
        torch.save(
            {"model_state_dict": model.state_dict(), "H": H, "W": W}, model_checkpoint
        )

    plot_results(train_losses, eval_losses, accuracies, steps, step)


def init_weights(m):
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


def evaluate(model, val_loader):
    model.eval()
    eval_loss, accuracy = eval_steps(model, val_loader)
    model.train()

    return eval_loss.cpu(), accuracy


def eval_steps(model, dataloader):
    with torch.no_grad():
        eval_loss = 0
        accuracy = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)
            eval_loss += loss.item()
            accuracy += calc_accuracy(output, labels)

        eval_loss = eval_loss / len(dataloader)
        accuracy = accuracy.item() / len(dataloader) * 100

    return loss, accuracy


def plot_results(train_losses, eval_losses, accuracies, steps, step):
    train_losses = np.array(train_losses)
    eval_losses = np.array(eval_losses)
    accuracies = np.array(accuracies)
    steps = np.array(steps)

    _, ax = plt.subplots()
    ax.plot(range(1, step + 1), train_losses, label="Train Loss", color="red")
    ax.plot(steps, eval_losses, label="Eval Loss", color="green")
    ax.set_xlabel("steps")
    ax.set_ylabel("loss")
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(steps, accuracies, label="Accuracy", color="blue")
    ax2.set_ylabel("accuracy (%)")
    ax2.legend()

    plt.savefig(visual_path + "/loss.png")


def calc_accuracy(output, labels):
    ps = torch.exp(output)
    _, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))

    return accuracy


if __name__ == "__main__":
    train()
