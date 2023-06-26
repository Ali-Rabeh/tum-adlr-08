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
    'shift_length': 1, 
    'sampling_frequency': 50, 
    'batch_size': 1, 

    'num_particles': 500, 
    'initial_covariance': torch.diag(torch.tensor([0.01, 0.01, 0.01])),
    'use_log_probs': True,

    'pretrain_forward_model': True,
    'save_model': False,

    'pretrain_epochs': 10, # will only be used if 'pretrain_forward_model' is set to True
    'epochs': 10,
    'learning_rate': 5e-5,

    'sequence_length': 4
}

def pretrain_forward_model_single_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    batch_losses = []
    for batch, (input_states, control_inputs, observations, target_states) in enumerate(dataloader):
        input_states, control_inputs, observations, target_states = input_states.to(device), control_inputs.to(device), observations.to(device), target_states.to(device)

        sequence_losses = []
        for t in range(input_states.shape[1]): 
            current_state = input_states[:,t,:].unsqueeze(dim=1)
            current_control = control_inputs[:,t,:]
            current_gt_state = target_states[:,t,:].unsqueeze(dim=1)

            if t % hparams['sequence_length'] == 0:
                loss = torch.zeros([]) 
                optimizer.zero_grad()
                pred = current_state

            pred = model.forward(pred, current_control)
            loss += loss_fn(pred, current_gt_state)

            if (t % hparams['sequence_length'] == hparams['sequence_length'] - 1) or (t == input_states.shape[1]-1): 
                loss.backward()
                optimizer.step()
                sequence_losses.append(loss)        

        batch_losses.append(torch.mean(torch.tensor(sequence_losses)))

    return model, torch.tensor(batch_losses)

def validate_forward_model_single_epoch(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for input_states, control_inputs, observations, target_states in dataloader:
            input_states, control_inputs, observations, target_states = input_states.to(device), control_inputs.to(device), observations.to(device), target_states.to(device)

            for t in range(input_states.shape[1]): 
                current_state = input_states[:,t,:].unsqueeze(dim=1)
                current_control = control_inputs[:,t,:]
                current_gt_state = target_states[:,t,:].unsqueeze(dim=1)

                if t % hparams['sequence_length'] == 0: 
                    pred = current_state

                pred = model.forward(pred, current_control)
                test_loss += loss_fn(pred, current_gt_state).item()

    test_loss /= num_batches 
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

        val_loss = validate_forward_model_single_epoch(validation_dataloader, model, loss_fn=loss_fn)
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
    best_model = None

    for epoch in range(hparams['epochs']): 
        batch_losses = []
        model.train() 

        # train for one epoch
        for batch, (input_states, control_inputs, observations, target_states) in enumerate(train_dataloader):
            input_states, control_inputs, observations, target_states = input_states.to(device), control_inputs.to(device), observations.to(device), target_states.to(device)

            sequence_losses = []
            for t in range(input_states.shape[1]):
                current_state = input_states[:,t,:].unsqueeze(dim=1)
                current_control = control_inputs[:,t,:]
                current_measurement = observations[:,t,:]
                current_gt_state = target_states[:,t,:]

                # if we start a new sequence, we have to zero everything and reinitialize the model
                if t % hparams['sequence_length'] == 0: 
                    loss = torch.zeros([])
                    optimizer.zero_grad()
                    model.initialize(current_state.shape[0], current_state, hparams['initial_covariance'])

                estimate = model.step(current_control, current_measurement)
                loss += loss_fn(estimate, current_gt_state)

                # backpropagate the loss at the end of a sequence
                if (t % hparams['sequence_length'] == hparams['sequence_length'] - 1) or (t == input_states.shape[1]-1):
                    loss.backward()
                    optimizer.step()
                    sequence_losses.append(loss)
                
            batch_losses.append(torch.mean(torch.tensor(sequence_losses)))
        train_losses.append(torch.mean(torch.tensor(batch_losses)))

        # validate the model
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for (input_states, control_inputs, observations, target_states) in validation_dataloader:
                input_states, control_inputs, observations, target_states = input_states.to(device), control_inputs.to(device), observations.to(device), target_states.to(device)

                for t in range(input_states.shape[1]): 
                    current_state = input_states[:,t,:].unsqueeze(dim=1)
                    current_control = control_inputs[:,t,:]
                    current_measurement = observations[:,t,:]
                    current_gt_state = target_states[:,t,:]                 

                    # if we start a new sequence, we have to reinitialize the PF
                    if t % hparams['sequence_length'] == 0:
                        model.initialize(current_state.shape[0], current_state, hparams['initial_covariance'])

                    estimate = model.step(current_control, current_measurement)
                    test_loss += loss_fn(estimate, current_gt_state).item()
                    # print(f"Validation: prediction = {estimate} | ground truth = {current_gt_state}")

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
        shift_length=hparams['shift_length'], 
        sampling_frequency=hparams['sampling_frequency']
        )
    train_dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=False)

    validation_dataset = SequenceDataset(
        validation_path_list,
        mode=hparams['mode'], 
        shift_length=hparams['shift_length'],
        sampling_frequency=hparams['sampling_frequency']
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=hparams['batch_size'], shuffle=False)

    # set the device used
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # define DPF
    dpf = DifferentiableParticleFilter(hparams, ForwardModel(), ObservationModel(use_log_probs=hparams['use_log_probs']))

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
        torch.save(best_dpf, "models/saved_models/20230609_PretrainEpochs100_Epochs200_SequenceLength4.pth")

if __name__ == '__main__':
    main()