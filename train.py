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

from run_filter import visualize_particles

hparams = {
    'mode': 'shifted', 
    'sequence_length': 2, 
    'sampling_frequency': 50, 
    'batch_size': 1, 

    'num_particles': 500, 
    'initial_covariance': torch.diag(torch.tensor([0.1, 0.1, 0.01])),

    'pretrain_epochs': 50, # will only be used if 'pretrain_forward_model' is set to True
    'epochs': 100,
    'learning_rate': 1e-4,

    'pretrain_forward_model': True,
    'save_model': False
}

def pretrain_forward_model_single_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    batch_losses = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.squeeze(), y.squeeze()
        X, y = X.to(device), y.to(device)
        states, control_inputs = X[:,7:10], X[:,4:7]

        # get the states into the correct shape for the forward model --> this is basically a distribution with just one particle
        states = torch.unsqueeze(states, dim=1)

        # Compute prediction error
        pred = model.forward(states, control_inputs)
        pred = pred.squeeze()
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_losses.append(loss)

        loss, current = loss.item(), (batch+1)
        # print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    return model, torch.tensor(batch_losses)

def validate_forward_model(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.squeeze().to(device), y.squeeze().to(device)
            states, control_inputs = X[:,7:10], X[:,4:7]
            states = torch.unsqueeze(states, dim=1)

            pred = model.forward(states, control_inputs)
            pred = pred.squeeze()
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches 
    # print(f"Validation loss: {test_loss:>8f} \n")

    return test_loss

def pretrain_forward_model(train_dataloader, validation_dataloader, model, loss_fn, optimizer): 
    """ Convenience function to handle both training and validation of the forward model pretraining.

    Args: 

    Returns: 
    
    """
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, validation_losses = [], []
    best_validation_loss = 1000.0
    for epoch in range(hparams['pretrain_epochs']):
        model, batch_losses = pretrain_forward_model_single_epoch(train_dataloader, model, loss_fn=loss_fn, optimizer=optimizer)
        train_losses.append(torch.mean(batch_losses))

        val_loss = validate_forward_model(validation_dataloader, model, loss_fn=loss_fn)
        validation_losses.append(val_loss)

        if val_loss < best_validation_loss:
            best_model = model

        print(f"Pretraining epoch {epoch+1} finished.")

    return best_model, train_losses, validation_losses

def train_end_to_end(train_dataloader, validation_dataloader, model, loss_fn, optimizer):
    """

    """
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, validation_losses = [], []
    best_validation_loss = 1000.0

    for epoch in range(hparams['epochs']): 
        batch_losses = []
        model.train() 

        # train for one epoch
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.squeeze(), y.squeeze()
            X, y = X.to(device), y.to(device)
            states, control_inputs, measurements = X[:,7:10], X[:,4:7], X[:,1:7]

            input_size = X.shape[0]
            model.initialize(input_size, states, hparams['initial_covariance']) # only valid for one step predictions!

            estimates = model.step(control_inputs, measurements)
            loss = loss_fn(estimates, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_losses.append(torch.mean(loss))

            loss, current = loss.item(), (batch+1)
            # print(f"Loss: {loss:>7f} [{current:>5d}/{len(train_dataloader):>5d}]")

        train_losses.append(torch.mean(torch.tensor(batch_losses)))

        # validate the model
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for (X, y) in validation_dataloader:
                X, y = X.squeeze(), y.squeeze()
                states, control_inputs, measurements = X[:,7:10], X[:,4:7], X[:,1:7]
                X, y = X.to(device), y.to(device)

                model.initialize(states.shape[0], states, hparams['initial_covariance']) # only valid for one step predictions
                pred = model.step(control_inputs, measurements)
                test_loss += loss_fn(pred, y).item()

        test_loss /= len(validation_dataloader) 
        validation_losses.append(test_loss)
        
        if test_loss <= best_validation_loss:
            best_model = model

        print(f"End-to-end epoch {epoch+1} finished.")

    return best_model, train_losses, validation_losses

def plot_losses(train_losses, validation_losses, title): 
    """

    """
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(train_losses)), train_losses, label='Train')
    ax.plot(range(len(validation_losses)), validation_losses, label='Validation')
    ax.legend()
    ax.grid()
    ax.set_title(title)
    return fig, ax

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

    # set the device used
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # define DPF
    dpf = DifferentiableParticleFilter(hparams, ForwardModel(), ObservationModel())

    # set up loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(dpf.parameters(), lr=hparams['learning_rate'])

    # pretrain if desired
    if hparams['pretrain_forward_model']: 
        forward_model_optimizer = torch.optim.Adam(dpf.forward_model.parameters(), lr=hparams['learning_rate']) # train only the forward model
        best_forward_model, train_losses, validation_losses = pretrain_forward_model(train_dataloader,validation_dataloader, dpf.forward_model, loss_fn, optimizer)

        dpf.forward_model = best_forward_model
        dpf.forward_model.requires_grad_(False) # freeze weights of the forward model if it has been pretrained

        fig_pretrain, _ = plot_losses(train_losses, validation_losses, "Losses pretraining")
        plt.show()

    # train end-to-end
    best_dpf, train_losses, validation_losses = train_end_to_end(train_dataloader, validation_dataloader, dpf, loss_fn, optimizer)
    fig_end_to_end, _ = plot_losses(train_losses, validation_losses, "Losses end-to-end")
    plt.show()

    # save the best model if desired
    if hparams['save_model']: 
        torch.save(best_dpf, "models/saved_models/20230608_Model01.pth")

if __name__ == '__main__':
    main()