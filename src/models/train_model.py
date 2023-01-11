import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import hydra
import logging
from torch.profiler import profile, ProfilerActivity

from model import ResNeStModel


'''
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
'''

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

@hydra.main(config_path="../../conf", config_name='config.yaml')
def train(config):
    hparams = config.experiment
    paths = config.paths

    logger = logging.getLogger(__name__)

    train_dataset = torch.load(paths.processed_data_path + "train_dataset.pt")
    val_dataset = torch.load(paths.processed_data_path + "val_dataset.pt")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=hparams.batch_size, shuffle=True
    )

    model = ResNeStModel()
    model.to(device)
    model.apply(init_weights)

    criterion = getattr(torch.nn, hparams.criterion)()
    optimizer = getattr(optim, hparams.optimizer)(model.parameters(), lr=hparams.lr, weight_decay=hparams.decay, momentum=hparams.momentum)
    scheduler = getattr(optim.lr_scheduler, hparams.scheduler)(optimizer, milestones=hparams.lr_epoch, gamma=hparams.lr_decay)

    train_losses, eval_losses, accuracies, steps = [], [], [], []
    train_loss = 0
    step = 0
    count = 0

    model.train()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=torch.profiler.schedule(skip_first=90, wait=70, warmup=10, active=20), record_shapes=True, profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(paths.profiles + hparams.name)) as prof:
        for e in range(hparams.epoch):
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                output = nn.LogSoftmax(dim=1)(model(images))
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_losses.append(train_loss / (step + 1))

                if step % hparams.eval_every == 0:
                    steps.append(step)

                    eval_loss, accuracy = evaluate(model, val_loader, criterion)

                    eval_losses.append(eval_loss)
                    accuracies.append(accuracy)

                    logger.info(
                        f"Epoch: {e}/{hparams.epoch}\tStep: {step}\tTrain Loss: {train_losses[-1]:.2f}\tEval Loss: {eval_losses[-1]:.2f}\tAccuracy: {accuracies[-1]:.2f}%"
                    )

                    if len(eval_losses) > 1 and eval_losses[-1] <= eval_losses[-2]:
                        count += 1
                    else:
                        count = 0

                    if count >= hparams.stop_after:
                        logger.info("It's time for early stopping. Let's save the model!")
                        step += 1
                        torch.save(model.state_dict(), paths.model_path + f'checkpoint_{hparams.name}.pth')
                        break
                
                step += 1
                prof.step()

            else:
                scheduler.step()
                continue

            break
        else:
            logger.info(
                f"Epoch: {e}/{hparams.epoch}\tStep: {step}\tTrain Loss: {train_losses[-1]:.2f}\tEval Loss: {eval_losses[-1]:.2f}\tAccuracy: {accuracies[-1]:.2f}%"
                )
            logger.info("Finish training and save the model.")
            torch.save(model.state_dict(), paths.model_path + f'checkpoint_{hparams.name}.pth')

    logger.info(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
    prof.export_chrome_trace(paths.profiles + f"{hparams.name}/trace.json")

    plot_results(train_losses, eval_losses, accuracies, steps, step, config)


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


def evaluate(model, val_loader, criterion):
    model.eval()
    eval_loss, accuracy = eval_steps(model, val_loader, criterion)
    model.train()

    return eval_loss, accuracy


def eval_steps(model, dataloader, criterion):
    with torch.no_grad():
        eval_loss = 0
        accuracy = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            output = nn.LogSoftmax(dim=1)(model(images))
            loss = criterion(output, labels)
            eval_loss += loss.item()
            accuracy += calc_accuracy(output, labels)

        eval_loss = eval_loss / len(dataloader)
        accuracy = accuracy.item() / len(dataloader) * 100

    return eval_loss, accuracy


def plot_results(train_losses, eval_losses, accuracies, steps, step, config):
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

    plt.savefig(config.paths.visual_path + f"loss_{config.experient.name}.png")


def calc_accuracy(output, labels):
    ps = torch.exp(output)
    _, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))

    return accuracy


if __name__ == "__main__":
    train()
