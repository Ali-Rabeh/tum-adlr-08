""" Helper script for calculating mean, std, min and max of the datasets for normalization.  

"""

import os
import torch
from torch.utils.data import DataLoader

from data_file_paths import train_path_list, validation_path_list, test_path_list
from dataset import SequenceDataset

hparams = {
    'mode': 'shifted',
    'shift_length': 2, 
    'sampling_frequency': 50, 
    'batch_size': 1
}

def calculate_mean_and_std(dataloader):
    """ Approximately calculates the mean and standard deviation of a dataset. """

    mean = torch.zeros(size=(1,12))
    std = torch.zeros(size=(1,12))
    number_of_samples = 0.0
    for batch, (input_states, control_inputs, observations, target_states) in enumerate(dataloader): 
        number_of_samples += input_states.shape[1]

        mean[:,0:3] += torch.mean(input_states, dim=1)
        std[:,0:3] += torch.std(input_states, dim=1)

        mean[:,3:6] += torch.mean(control_inputs, dim=1)
        std[:,3:6] += torch.std(control_inputs, dim=1)

        mean[:,6:9] += torch.mean(observations, dim=1)
        std[:,6:9] += torch.std(observations, dim=1)

        mean[:,9:12] += torch.mean(target_states, dim=1)
        std[:,9:12] += torch.std(target_states, dim=1)

    mean = mean / number_of_samples
    std = std / number_of_samples
    return mean, std

def calculate_min_max_of_dataset(dataloader):
    """
    
    """
    min_values = 10*torch.ones(size=(1,12))
    max_values = -10* torch.ones(size=(1,12))
    # for batch, (input_states, control_inputs, observations, target_states) in enumerate(dataloader): 
    for batch, modality in enumerate(dataloader):
        for counter in range(len(modality)): 

            min_values_batch = torch.min(modality[counter], dim=1).values
            replacement_indices = min_values_batch < min_values[:,counter*(len(modality)-1):(counter+1)*(len(modality)-1)]
            min_values[:,counter*(len(modality)-1):(counter+1)*(len(modality)-1)][replacement_indices] = torch.min(modality[counter], dim=1).values[replacement_indices]

            max_values_batch = torch.max(modality[counter], dim=1).values
            replacement_indices = max_values_batch > min_values[:,counter*(len(modality)-1):(counter+1)*(len(modality)-1)]
            max_values[:,counter*(len(modality)-1):(counter+1)*(len(modality)-1)][replacement_indices] = torch.max(modality[counter], dim=1).values[replacement_indices] 

    return min_values, max_values    

def main(): 
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

    test_dataset = SequenceDataset(
        test_path_list, 
        mode=hparams['mode'], 
        sampling_frequency = hparams["sampling_frequency"],
        shift_length=hparams["shift_length"]
    )
    test_dataloader = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False)

    # calculate mean and standard deviation
    train_mean, train_std = calculate_mean_and_std(train_dataloader)
    train_min, train_max = calculate_min_max_of_dataset(train_dataloader)

    val_mean, val_std = calculate_mean_and_std(validation_dataloader)
    val_min, val_max = calculate_min_max_of_dataset(validation_dataloader)

    test_mean, test_std = calculate_mean_and_std(test_dataloader)
    test_min, test_max = calculate_min_max_of_dataset(test_dataloader)

    # print(f"Train set: mean = {train_mean} | std = {train_std}")
    # print(f"Validation set: mean = {val_mean} | std = {val_std}")
    # print(f"Test set: mean = {test_mean} | std = {test_std}")

    print(f"Train set: min = {train_min} | max = {train_max}")
    print(f"Validation set: min = {val_min} | max = {val_max}")
    print(f"Test set: min = {test_min} | max = {test_max}")

if __name__ == "__main__": 
    main()