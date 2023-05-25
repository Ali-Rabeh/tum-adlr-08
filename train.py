import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from models.forward_model import ForwardModel
from util.dataset import NormalDataset, ShiftedDataset, assemble_datasets

hparams = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'epochs': 20,
    'sequence_length': 2
}

# standard train function from pytorch tutorials
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    batch_losses = []
    for batch, (X, y) in enumerate(dataloader):
        X = X[:,7:10]
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_losses.append(loss)

        if batch % 10 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    return torch.tensor(batch_losses)

# standard test function from pytorch tutorials
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X[:,7:10]
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches 
    print(f"Test Error: Avg. loss: {test_loss:>8f} \n")

    return test_loss

if __name__ == '__main__':
    # build training dataloader
    train_path_list = [os.path.abspath("../pushdata/plywood/butter/motion_surface=plywood_shape=butter_a=0_v=10_i=0.000_s=0.000_t=0.000.json"),
                       os.path.abspath("../pushdata/plywood/butter/motion_surface=plywood_shape=butter_a=0_v=10_i=0.000_s=0.000_t=0.698.json")]
    training_data = assemble_datasets(train_path_list, mode='shifted', sequence_length=hparams['sequence_length'])
    train_dataloader = DataLoader(training_data, batch_size=hparams['batch_size'], shuffle=False, drop_last=True)
    print(f"Length training dataloader: {len(train_dataloader)}")

    # build validation dataloader
    validation_path_list = [os.path.abspath("../pushdata/plywood/butter/motion_surface=plywood_shape=butter_a=0_v=10_i=0.000_s=0.000_t=0.349.json")]
    validation_data = assemble_datasets(validation_path_list, mode='shifted', sequence_length=hparams['sequence_length'])
    validation_dataloader = DataLoader(validation_data, batch_size=hparams['batch_size'], shuffle=False, drop_last=True)
    print(f"Length validation dataloader: {len(validation_dataloader)}")

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # build the model
    model = ForwardModel().to(device)
    print(model)

    # prepare training
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])

    epochs = hparams['epochs']
    train_epoch_losses, validation_epoch_losses = [], []
    for t in range(epochs):
        print(f"Epoch {t+1}\n---------------------------------------")

        train_batch_losses = train(train_dataloader, model, loss_fn, optimizer)
        train_epoch_losses.append(torch.mean(train_batch_losses))

        validation_loss = test(validation_dataloader, model, loss_fn)
        validation_epoch_losses.append(validation_loss)

    print("Done!")

    fig = plt.figure()
    plt.plot(range(epochs), train_epoch_losses)
    plt.plot(range(epochs), validation_epoch_losses)
    plt.legend(["Train loss", "Validation loss"])
    plt.show()

