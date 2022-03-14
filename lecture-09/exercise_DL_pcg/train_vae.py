"""
This script trains a VAESimple on the Mario
levels.
"""
import json
from time import time

import click
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm

from vae_mario import VAEMario

# Data types.
Tensor = torch.Tensor


def load_data(training_percentage=0.8, test_percentage=None, shuffle_seed=0):
    """Returns two tensors with training and testing data"""
    # Loading the data.
    # This data is structured [b, c, i, j], where c corresponds to the class.
    data = np.load("./all_levels_onehot.npz")["levels"]

    np.random.seed(shuffle_seed)
    np.random.shuffle(data)

    # Separating into training and test.
    n_data, _, _, _ = data.shape
    training_index = int(n_data * training_percentage)
    training_data = data[:training_index, :, :, :]
    testing_data = data[training_index:, :, :, :]
    training_tensors = torch.from_numpy(training_data).type(torch.float)
    test_tensors = torch.from_numpy(testing_data).type(torch.float)

    return training_tensors, test_tensors


def loss_function(x_prime, x, mu, log_var, scale=1.0):
    x_classes = x.argmax(dim=1)
    loss = torch.nn.NLLLoss(reduction="sum")
    CEL = loss(x_prime, x_classes)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return CEL + scale * KLD, CEL, KLD


def fit(model, optimizer, data_loader, device, scale=1.0):
    model.train()
    running_loss = 0.0
    for _, levels in tqdm(enumerate(data_loader)):
        levels = levels[0]
        levels = levels.to(device)
        optimizer.zero_grad()
        x_primes, xs, mu, log_var = model(levels)
        loss, _, _ = loss_function(x_primes, xs, mu, log_var, scale=scale)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    return running_loss


def test(model, test_loader, test_dataset, device, epoch=0, scale=1.0):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for _, levels in tqdm(enumerate(test_loader)):
            levels = levels[0]
            levels.to(device)
            x_primes, xs, mu, log_var = model(levels)
            loss, _, _ = loss_function(x_primes, xs, mu, log_var, scale=scale)
            running_loss += loss.item()

    print(f"Epoch {epoch}. Loss in test: {running_loss / len(test_dataset)}")
    return running_loss


@click.command()
@click.option("--z-dim", type=int, default=2)
@click.option("--max-epochs", type=int, default=200)
@click.option("--batch-size", type=int, default=64)
@click.option("--lr", type=float, default=1e-3)
@click.option("--seed", type=int, default=0)
@click.option("--save-every", type=int, default=20)
@click.option("--overfit/--no-overfit", default=False)
def run(
    z_dim,
    max_epochs,
    batch_size,
    lr,
    seed,
    save_every,
    overfit,
):
    # Setting up the seeds
    torch.manual_seed(seed)

    # Setting up the hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the data.
    training_tensors, test_tensors = load_data(shuffle_seed=seed)

    # Creating datasets.
    dataset = TensorDataset(training_tensors)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_tensors)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # Loading the model
    vae = VAEMario(z_dim=z_dim)

    optimizer = optim.Adam(vae.parameters(), lr=lr)

    best_loss = np.Inf
    n_without_improvement = 0
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1} of {max_epochs}.")
        train_loss = fit(vae, optimizer, data_loader, device)
        test_loss = test(vae, test_loader, test_dataset, device, epoch=epoch)

        if test_loss < best_loss:
            best_loss = test_loss
            n_without_improvement = 0

            # Saving the best model so far.
            torch.save(vae.state_dict(), f"./models/mario_vae_zdim_{z_dim}_final.pt")
        else:
            if not overfit:
                n_without_improvement += 1

        if epoch % save_every == 0 and epoch != 0:
            # Saving the model
            print(f"Saving the model at checkpoint {epoch}.")
            torch.save(
                vae.state_dict(), f"./models/mario_vae_zdim_{z_dim}_epoch_{epoch}.pt"
            )

        # Early stopping:
        if n_without_improvement == 10:
            print("Stopping early")
            break


if __name__ == "__main__":
    run()
