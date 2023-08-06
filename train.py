import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from models.forward_model import ForwardModel
from models.observation_model import ObservationModel, ObservationModelImages
from models.differentiable_particle_filter import DifferentiableParticleFilter

from util.manifold_helpers import boxplus, radianToContinuous, continuousToRadian
from util.data_file_paths import *
from util.dataset import SequenceDataset

from run_filter import visualize_particles, unnormalize_min_max

from camera_helpers import ImageGenerator

hparams = {
    'mode': 'shifted', 
    'shift_length': 1, 
    'sampling_frequency': 50, 
    'batch_size': 1, 

    'use_forces': False, 
    'use_images': True,

    'num_particles': 100, 
    'initial_covariance': torch.diag(torch.tensor([0.2, 0.2, 0.2])),
    'use_log_probs': True,
    'use_resampling': True,
    'resampling_soft_alpha': 0.05,

    'lr_decay_rate': 0.95,
    'early_stopping_epochs': 10,

    'pretrain_forward_model': False,
    'use_pretrained_forward_model': True,

    'save_model': True,
    'model_name': "20230806_OnlyImageObservationsWithPretrainedForwardModel.pth",

    'pretrain_epochs': [10, 10, 20, 20, 50], # will only be used if 'pretrain_forward_model' is set to True
    'epochs': [50, 50, 50, 50, 50],
    'learning_rate': 1e-4,

    'sequence_lengths': [1, 2, 4, 8, 16]
}

# prepare image generation
image_generator = ImageGenerator()

def pretrain_forward_model_single_epoch(dataloader, model, loss_fn, optimizer, sequence_length):
    size = len(dataloader.dataset)
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    batch_losses = []
    for batch, (input_states, control_inputs, observations, target_states) in enumerate(dataloader):
        input_states, control_inputs, observations, target_states = input_states.to(device), control_inputs.to(device), observations.to(device), target_states.to(device)

        sequence_losses = []
        for t in range(input_states.shape[1]): 
            current_state = input_states[:,t,:].unsqueeze(dim=1)
            current_control = control_inputs[:,t,:].unsqueeze(dim=1)
            current_gt_state = target_states[:,t,:].unsqueeze(dim=1)

            # print(f"Current state: {current_state}")
            # print(f"Current control: {current_control}")
            # print(f"Current GT state: {current_gt_state}")

            if t % sequence_length == 0:
                loss = torch.zeros([]) 
                optimizer.zero_grad()
                pred_state = current_state

            pred_state = radianToContinuous(pred_state)
            current_control = radianToContinuous(current_control)

            pred_delta = model.forward(pred_state, current_control)
            # print(f"Pred. state: {pred_state.shape} | Pred. delta: {pred_delta.shape}")

            pred_delta = continuousToRadian(pred_delta)
            pred_state = continuousToRadian(pred_state)

            pred_state = boxplus(pred_state, pred_delta)
            loss += loss_fn(pred_state, current_gt_state)

            if (t % sequence_length == sequence_length - 1) or (t == input_states.shape[1]-1): 
                loss.backward()
                optimizer.step()
                sequence_losses.append(loss)        

        batch_losses.append(torch.mean(torch.tensor(sequence_losses)))

    return model, torch.tensor(batch_losses)

def validate_forward_model_single_epoch(dataloader, model, loss_fn, sequence_length):
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
                current_control = control_inputs[:,t,:].unsqueeze(dim=1)
                current_gt_state = target_states[:,t,:].unsqueeze(dim=1)

                if t % sequence_length == 0: 
                    pred_state = current_state

                pred_state = radianToContinuous(pred_state)
                current_control = radianToContinuous(current_control)
                pred_delta = model.forward(pred_state, current_control)
                # print(f"Pred delta directly from model: {pred_delta}")

                pred_delta = continuousToRadian(pred_delta)
                pred_state = continuousToRadian(pred_state)
                # print(f"Pred delta: {pred_delta} | Pred state: {pred_state}")

                pred_state = boxplus(pred_state, pred_delta)
                test_loss += loss_fn(pred_state, current_gt_state).item()

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
    for sequence_index in range(len(hparams['sequence_lengths'])):
        sequence_length = hparams['sequence_lengths'][sequence_index]
        best_validation_loss = 1000.0

        for epoch in range(hparams['pretrain_epochs'][sequence_index]):
            learning_rate = hparams['lr_decay_rate'] ** epoch * hparams['learning_rate']
            optimizer.param_groups[0]['lr'] = learning_rate

            model, batch_losses = pretrain_forward_model_single_epoch(train_dataloader, model, loss_fn=loss_fn, optimizer=optimizer, sequence_length=sequence_length)
            train_losses.append(torch.mean(batch_losses))

            val_loss = validate_forward_model_single_epoch(validation_dataloader, model, loss_fn=loss_fn, sequence_length=sequence_length)
            validation_losses.append(val_loss)

            # from the last sequence, that is, the longest sequence of prediction steps used for training, we want the best model
            if val_loss < best_validation_loss and sequence_index==len(hparams['sequence_lengths'])-1:
                best_validation_loss = val_loss
                best_model = model

            print("Pretraining | Sequence length = {} | LR = {:.4e} | Epoch {} finished with validation loss: {}".format(
                    sequence_length, 
                    optimizer.param_groups[0]['lr'],
                    epoch+1,
                    val_loss
                )
            )

    return best_model, train_losses, validation_losses

def train_end_to_end_single_epoch(dataloader, model, loss_fn, optimizer, sequence_length): 
    """

    """
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    batch_losses = []
    for batch, (input_states, control_inputs, observations, target_states) in enumerate(dataloader):
        input_states, control_inputs, observations, target_states = input_states.to(device), control_inputs.to(device), observations.to(device), target_states.to(device)

        sequence_losses = []
        for t in range(input_states.shape[1]):
            current_state = input_states[:,t,:].unsqueeze(dim=1)
            current_control = control_inputs[:,t,:].unsqueeze(dim=1)
            current_measurement = observations[:,t,:]
            current_gt_state = target_states[:,t,:]

            current_image = image_generator.generate_image(
                unnormalize_min_max(current_gt_state.squeeze(), train_min[:,0:3].squeeze(), train_max[:,0:3].squeeze())
            )
            current_image = torch.tensor(current_image[None, None, :, :], dtype=torch.float32) / 255.0

            # if we start a new sequence, we have to zero everything and reinitialize the model
            if t % sequence_length == 0: 
                loss = torch.zeros([])
                optimizer.zero_grad()
                model.initialize(current_state.shape[0], current_state, hparams['initial_covariance'])

            if hparams['use_forces'] and not hparams['use_images']:
                estimate = model.step(current_control, measurement=current_measurement)
            if not hparams['use_forces'] and hparams['use_images']:
                estimate = model.step(current_control, image=current_image)

            loss += loss_fn(estimate, current_gt_state)

            # backpropagate the loss at the end of a sequence
            if (t % sequence_length == sequence_length - 1) or (t == input_states.shape[1]-1):
                loss.backward()
                optimizer.step()
                sequence_losses.append(loss)
            
        batch_losses.append(torch.mean(torch.tensor(sequence_losses)))
    return model, batch_losses

def validate_end_to_end_single_epoch(dataloader, model, loss_fn, sequence_length):
    """

    """
    model.eval()
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    val_loss = 0
    with torch.no_grad():
        for (input_states, control_inputs, observations, target_states) in dataloader:
            input_states, control_inputs, observations, target_states = input_states.to(device), control_inputs.to(device), observations.to(device), target_states.to(device)

            for t in range(input_states.shape[1]): 
                current_state = input_states[:,t,:].unsqueeze(dim=1)
                current_control = control_inputs[:,t,:].unsqueeze(dim=1)
                current_measurement = observations[:,t,:]
                current_gt_state = target_states[:,t,:]                 

                current_image = image_generator.generate_image(
                    unnormalize_min_max(current_gt_state.squeeze(), validation_min[:,0:3].squeeze(), validation_max[:,0:3].squeeze())
                )
                current_image = torch.tensor(current_image[None, None, :, :], dtype=torch.float32)

                # if we start a new sequence, we have to reinitialize the PF
                if t % sequence_length == 0:
                    model.initialize(current_state.shape[0], current_state, hparams['initial_covariance'])

                if hparams['use_forces'] and not hparams['use_images']: 
                    estimate = model.step(current_control, measurement=current_measurement)
                if not hparams['use_forces'] and hparams['use_images']: 
                    estimate = model.step(current_control, image=current_image)

                val_loss += loss_fn(estimate, current_gt_state).item()
                # print(f"Validation: input state: {current_state} | prediction = {estimate} | ground truth = {current_gt_state}")

    val_loss /= len(dataloader)
    return val_loss

def train_end_to_end(train_dataloader, validation_dataloader, model, loss_fn, optimizer):
    """

    """
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, validation_losses = [], []
    best_model = None

    for sequence_index in range(len(hparams['sequence_lengths'])):
        sequence_length = hparams['sequence_lengths'][sequence_index]
        best_validation_loss = 100000.0
        early_stopping_counter = 0
        for epoch in range(hparams['epochs'][sequence_index]):
            learning_rate = hparams['lr_decay_rate'] ** epoch * hparams['learning_rate']
            optimizer.param_groups[0]['lr'] = learning_rate            

            model, batch_losses = train_end_to_end_single_epoch(train_dataloader, model, loss_fn, optimizer, sequence_length)
            train_losses.append(torch.mean(torch.tensor(batch_losses)))

            val_loss = validate_end_to_end_single_epoch(validation_dataloader, model, loss_fn, sequence_length)
            validation_losses.append(val_loss)

            if val_loss <= best_validation_loss:
                best_validation_loss = val_loss
                early_stopping_counter = 0 

                if sequence_index==len(hparams['sequence_lengths'])-1:
                    best_model = model
            else:
                early_stopping_counter += 1

            print("End-to-end | Sequence length = {} | LR = {:.4e} | Epoch = {} | Train loss: {:.6f} | Val. loss: {:.6f} | {}".format(
                    sequence_length, 
                    optimizer.param_groups[0]['lr'],
                    epoch+1,
                    torch.mean(torch.tensor(batch_losses)),
                    val_loss,
                    early_stopping_counter
                )
            )
            
            if early_stopping_counter == hparams['early_stopping_epochs']: 
                print("Early stopping.")
                break

    return best_model, train_losses, validation_losses

def plot_losses(train_losses, validation_losses, title): 
    """

    """
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(train_losses)), np.log(train_losses), label='Train')
    ax.plot(range(len(validation_losses)), np.log(validation_losses), label='Validation')

    ax.set_title(title)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Log-Loss")

    ax.legend()
    ax.grid()
    return fig, ax

def main(): 
    # build up training dataset and dataloader
    train_dataset = SequenceDataset(
        train_path_list, 
        mode=hparams['mode'], 
        shift_length=hparams['shift_length'], 
        sampling_frequency=hparams['sampling_frequency'],
        dataset_min=train_min,
        dataset_max=train_max
        )
    train_dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=False)

    validation_dataset = SequenceDataset(
        validation_path_list,
        mode=hparams['mode'], 
        shift_length=hparams['shift_length'],
        sampling_frequency=hparams['sampling_frequency'],
        dataset_min=validation_min,
        dataset_max=validation_max
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=hparams['batch_size'], shuffle=False)

    # set the device used
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # define DPF
    if hparams['use_forces'] and not hparams['use_images']: 
        observation_model = ObservationModel(use_log_probs=hparams['use_log_probs'])
    if not hparams['use_forces'] and hparams['use_images']: 
        observation_model = ObservationModelImages(use_log_probs=hparams['use_log_probs'])

    dpf = DifferentiableParticleFilter(hparams, ForwardModel(), observation_model)

    # set up loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(dpf.parameters(), lr=hparams['learning_rate'])

    # pretrain if desired
    if hparams['pretrain_forward_model']: 
        forward_model_optimizer = torch.optim.Adam(dpf.forward_model.parameters(), lr=hparams['learning_rate']) # train only the forward model
        best_forward_model, train_losses, validation_losses = pretrain_forward_model(train_dataloader,validation_dataloader, dpf.forward_model, loss_fn, optimizer)

        dpf.forward_model = best_forward_model
        # torch.save(best_forward_model, 'experiments/20230806_SimpleForwardModel_UpTo16Steps.pth') 

        fig_pretrain, _ = plot_losses(train_losses, validation_losses, "Losses pretraining")
        plt.show()

    if hparams['use_pretrained_forward_model']: 
        dpf.forward_model = torch.load('experiments/20230806_SimpleForwardModel_UpTo16Steps.pth')
        dpf.forward_model.requires_grad_(False) # freeze weights of the forward model if it has been pretrained

    # train end-to-end
    best_dpf, train_losses, validation_losses = train_end_to_end(train_dataloader, validation_dataloader, dpf, loss_fn, optimizer)
    fig_end_to_end, _ = plot_losses(train_losses, validation_losses, "Losses end-to-end")
    plt.show()

    # save the best model if desired
    print(best_dpf.parameters())
    if hparams['save_model']: 
        torch.save(best_dpf, "models/saved_models/"+hparams['model_name'])

if __name__ == '__main__':
    main()