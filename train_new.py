import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from models.forward_model import ForwardModel
from models.observation_model import ObservationModel
from models.differentiable_particle_filter import DifferentiableParticleFilter

from util.data_file_paths import train_path_list, validation_path_list
from util.dataset import SequenceDataset

hparams = {
    'mode': 'shifted', 
    'sequence_length': 2, 
    'sampling_frequency': 50, 
    'batch_size': 1, 

    'num_particles': 500, 
    'initial_covariance': torch.diag(torch.tensor([0.1, 0.1, 0.01])),

    'epochs': 100,
    'learning_rate': 1e-4
}

def train_forward_model(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    batch_losses = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.squeeze(), y.squeeze()
        X, y = X.to(device), y.to(device)

        states = X[:,7:10]
        control_inputs = X[:,4:7]

        # Compute prediction error
        pred = model(states, control_inputs)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_losses.append(loss)

        loss, current = loss.item(), (batch+1)
        print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    return torch.tensor(batch_losses)

def validate_forward_model(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.squeeze(), y.squeeze()
            states, control_inputs = X[:,7:10], X[:,4:7]
            X, y = X.to(device), y.to(device)
            pred = model(states, control_inputs)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches 
    print(f"Test Error: Avg. loss: {test_loss:>8f} \n")

    return test_loss

def train_dpf(dataloader, model, loss_fn, optimizer):
 return NotImplementedError

def main(): 
    # build up training dataset and dataloader
    train_dataset = SequenceDataset(
        train_path_list, 
        mode=hparams['mode'], 
        sequence_length=hparams['sequence_length'], 
        sampling_frequency=hparams['sampling_frequency']
        )
    train_dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=False)

    validation_dataset = SequenceDataset(
        validation_path_list,
        mode=hparams['mode'], 
        sequence_length=hparams['sequence_length'],
        sampling_frequency=hparams['sampling_frequency']
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=hparams['batch_size'], shuffle=False)

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # define DPF
    dpf = DifferentiableParticleFilter(hparams, ForwardModel(), ObservationModel())

    # model = dpf.forward_model
    # model.to(device)

    # epoch_losses, validation_losses = [], []
    # for epoch in range(hparams['epochs']):
    #     batch_losses = train_forward_model(train_dataloader, model, loss_fn=nn.MSELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=hparams['learning_rate']))
    #     epoch_losses.append(torch.mean(batch_losses))

    #     val_loss = validate_forward_model(validation_dataloader, model, loss_fn=nn.MSELoss())
    #     validation_losses.append(val_loss)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(dpf.parameters(), lr=hparams['learning_rate'])

    dpf.to(device)
    dpf.train()
    epoch_losses = []
    for epoch in range(hparams['epochs']): 
        batch_losses = []
        for batch, (X, y) in enumerate(train_dataloader): 
            X, y = X.squeeze(), y.squeeze()
            states, control_inputs, measurements = X[:,7:10], X[:,4:7], X[:,1:7]

            input_size = X.shape[0]
            optimizer.zero_grad()
            dpf.initialize(input_size, states, hparams['initial_covariance']) # only valid for one step predictions!

            estimates = dpf.step(states, control_inputs, measurements)
            loss = loss_fn(estimates, y)

            loss.backward()
            optimizer.step()

            batch_losses.append(torch.mean(loss))

            loss, current = loss.item(), (batch+1)
            # print(f"Loss: {loss:>7f} [{current:>5d}/{len(train_dataloader):>5d}]")

        epoch_losses.append(torch.mean(torch.tensor(batch_losses)))
        print(f"Epoch {epoch+1} finished.")

    fig, ax = plt.subplots(1, 1)
    ax.plot(range(hparams['epochs']), epoch_losses, label='Train')
    # ax.plot(range(hparams['epochs']), validation_losses, label='Validation')
    ax.legend()
    ax.grid()
    ax.set_title('Loss')
    plt.show()

if __name__ == '__main__':
    main()