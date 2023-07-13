"""

"""

import torch
from torch.utils.data import Dataset, ConcatDataset

import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as pyplot

def assemble_datasets(path_list, mode, sampling_frequency, shift_length=1):
    """Concatenates indivual datasets when multiple paths are given. 

    Args: 
        path_list (list of strings): Valid paths to multiple json files. Each entry in the list should correspond to a path to single json file.

        mode (string): Either "normal" for chaining together NormalDatasets or "shifted" for chaining together ShiftedDatasets

        shift_length (int): Only relevant when specifying mode="shifted". See the docstring of ShiftedDataset for more information. 

    Returns:
        dataset (ConcatDataset): a dataset incorporating all the information from the json files given in path_list. 

    """
    list_of_datasets = []
    for path in path_list:
        if mode == 'shifted':
            list_of_datasets.append(ShiftedDataset(path, shift_length, sampling_rate=sampling_frequency))
        elif mode == 'normal':
            list_of_datasets.append(NormalDataset(path, sampling_rate=sampling_frequency))
        
    dataset = ConcatDataset(list_of_datasets)
    return dataset

class NormalDataset(Dataset):
    """Gets the data from a json file from the MIT Push dataset into a pytorch dataset format. 

    Args: 
        path (string): Valid path to a single json file.
    """
    def __init__(self, path, sampling_rate=-1):
        self.path = path

        with open(path) as data_file:
            data = json.load(data_file)
        help_df = pd.json_normalize(data)

        self.tip_pose_df = pd.DataFrame(
                        help_df.tip_pose[0], 
                        index=range(len(help_df.tip_pose[0])),
                        columns=['time_sec', 'tip_pose_xpos_m', 'tip_pose_ypos_m', 'tip_pose_theta_pos_rad']
                        )

        self.obj_pose_df = pd.DataFrame(
                        help_df.object_pose[0],
                        index = range(len(help_df.object_pose[0])),
                        columns=['time_sec', 'obj_pose_xcenter_m', 'obj_pose_ycenter_m', 'obj_pose_theta_rad']
                        )

        self.ft_wrench_df = pd.DataFrame(
                        help_df.ft_wrench[0],
                        index=range(len(help_df.ft_wrench[0])),
                        columns=['time_sec', 'ft_wrench_xforce_N', 'ft_wrench_yforce_N', 'ft_wrench_ztorque_Nm']
                        )

        # synchronize to the timestamps of the tip pose
        self.ft_wrench_df = self.ft_wrench_df.sort_values('time_sec')
        self.dataframe = pd.merge_asof(self.tip_pose_df, self.ft_wrench_df, on='time_sec', direction='nearest')
        self.obj_pose_df = self.obj_pose_df.sort_values('time_sec')
        self.dataframe = pd.merge_asof(self.dataframe, self.obj_pose_df, on='time_sec', direction='nearest')

        # set the initial object pose to (0, 0, 0)
        self.dataframe.loc[:, 'obj_pose_xcenter_m'] = self.dataframe.loc[:, 'obj_pose_xcenter_m'] - self.dataframe.loc[0, 'obj_pose_xcenter_m']
        self.dataframe.loc[:, 'obj_pose_ycenter_m'] = self.dataframe.loc[:, 'obj_pose_ycenter_m'] - self.dataframe.loc[0, 'obj_pose_ycenter_m']
        self.dataframe.loc[:, 'obj_pose_theta_rad'] = self.dataframe.loc[:, 'obj_pose_theta_rad'] - self.dataframe.loc[0, 'obj_pose_theta_rad']

        # resample if resampling frequency has been given
        if sampling_rate != -1:
            base_frequency = 250.0
            target_frequency = sampling_rate

            number_of_target_samples = int(target_frequency / base_frequency * len(self.dataframe))
            target_indices = np.linspace(0, len(self.dataframe)-1, num=number_of_target_samples, dtype=int)
            self.dataframe = self.dataframe.iloc[target_indices,:].reset_index()
            self.dataframe = self.dataframe.drop(columns='index')            

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index): 
        X = torch.tensor(self.dataframe.iloc[index,:], dtype=torch.float32)
        y = torch.tensor(self.dataframe.iloc[index,7:10], dtype=torch.float32)
        return X, y 

class ShiftedDataset(Dataset):
    """A dataset where the targets (in this case the true object pose) are shifted by (sequence_length-1) steps into the future, 
    to get a temporal sequence for predictions.

    Args:
        path_to_file (string): Valid path to a single json file.

        sequence_length (int): The targets (ground truth object poses) will be shifted by (sequence_length-1) timesteps into the future.
    """
    def __init__(self, path_to_file, shift_length, sampling_rate=-1):
        self.path_to_file = path_to_file
        self.shift_length = shift_length

        with open(path_to_file) as data_file:
            data = json.load(data_file)
        help_df = pd.json_normalize(data)

        self.tip_pose_df = pd.DataFrame(
                        help_df.tip_pose[0], 
                        index=range(len(help_df.tip_pose[0])),
                        columns=['time_sec', 'tip_pose_xpos_m', 'tip_pose_ypos_m', 'tip_pose_theta_pos_rad']
                        )
        pos_mask = self.tip_pose_df['tip_pose_theta_pos_rad'] > np.pi
        neg_mask = self.tip_pose_df['tip_pose_theta_pos_rad'] < -np.pi
        if pos_mask.any():
            print(f"Tip poses: Found angles >pi.")
        if neg_mask.any():
            print(f"Tip poses: Found angles <pi.")


        self.obj_pose_df = pd.DataFrame(
                        help_df.object_pose[0],
                        index = range(len(help_df.object_pose[0])),
                        columns=['time_sec', 'obj_pose_xcenter_m', 'obj_pose_ycenter_m', 'obj_pose_theta_rad']
                        )
        pos_mask = self.obj_pose_df['obj_pose_theta_rad'] > np.pi
        neg_mask = self.obj_pose_df['obj_pose_theta_rad'] < -np.pi
        if pos_mask.any():
            print(f"Object poses: Found angles >pi.")
        if neg_mask.any():
            print(f"Object poses: Found angles <pi.")

        self.ft_wrench_df = pd.DataFrame(
                        help_df.ft_wrench[0],
                        index=range(len(help_df.ft_wrench[0])),
                        columns=['time_sec', 'ft_wrench_xforce_N', 'ft_wrench_yforce_N', 'ft_wrench_ztorque_Nm']
                        )
                        
        # synchronize to the timestamps of the tip pose
        self.ft_wrench_df = self.ft_wrench_df.sort_values('time_sec')
        self.dataframe = pd.merge_asof(self.tip_pose_df, self.ft_wrench_df, on='time_sec', direction='nearest')
        self.obj_pose_df = self.obj_pose_df.sort_values('time_sec')
        self.dataframe = pd.merge_asof(self.dataframe, self.obj_pose_df, on='time_sec', direction='nearest')

        # set the initial object pose to (0, 0, 0)
        self.dataframe.loc[:, 'obj_pose_xcenter_m'] = self.dataframe.loc[:, 'obj_pose_xcenter_m'] - self.dataframe.loc[0, 'obj_pose_xcenter_m']
        self.dataframe.loc[:, 'obj_pose_ycenter_m'] = self.dataframe.loc[:, 'obj_pose_ycenter_m'] - self.dataframe.loc[0, 'obj_pose_ycenter_m']
        self.dataframe.loc[:, 'obj_pose_theta_rad'] = self.dataframe.loc[:, 'obj_pose_theta_rad'] - self.dataframe.loc[0, 'obj_pose_theta_rad']

        # resample the data to target sampling rate
        if sampling_rate != -1:
            base_frequency = 250.0
            target_frequency = sampling_rate

            number_of_target_samples = int(target_frequency / base_frequency * len(self.dataframe))
            target_indices = np.linspace(0, len(self.dataframe)-1, num=number_of_target_samples, dtype=int)
            self.dataframe = self.dataframe.iloc[target_indices,:].reset_index()
            self.dataframe = self.dataframe.drop(columns='index')

        # calculate the deltas of the object pose
        self.dataframe['delta_obj_pose_xcenter_m'] = self.dataframe['obj_pose_xcenter_m'].shift(+1) - self.dataframe['obj_pose_xcenter_m']
        self.dataframe['delta_obj_pose_ycenter_m'] = self.dataframe['obj_pose_ycenter_m'].shift(+1) - self.dataframe['obj_pose_ycenter_m']
        self.dataframe['delta_obj_pose_theta_rad'] = self.dataframe['obj_pose_theta_rad'].shift(+1) - self.dataframe['obj_pose_theta_rad']

        # filter out any rows with NaNs that have been produced by the shifting
        self.dataframe.dropna(axis=0, how='any', inplace=True)
        # print(self.dataframe.head())

    def __len__(self):
        return len(self.dataframe) - (self.shift_length)

    def __getitem__(self, index):
        # shift the output index to get a prediction
        input_index = index
        output_index = index+(self.shift_length)

        # X = torch.tensor(self.dataframe.iloc[input_index,:], dtype=torch.float32)
        # y = torch.tensor(self.dataframe.iloc[output_index,7:10], dtype=torch.float32)

        input_states = torch.tensor([self.dataframe.loc[input_index, 'obj_pose_xcenter_m'],
                                     self.dataframe.loc[input_index, 'obj_pose_ycenter_m'], 
                                     self.dataframe.loc[input_index, 'obj_pose_theta_rad']])
        control_inputs = torch.tensor([self.dataframe.loc[input_index, 'delta_obj_pose_xcenter_m'], 
                                       self.dataframe.loc[input_index, 'delta_obj_pose_ycenter_m'], 
                                       self.dataframe.loc[input_index, 'delta_obj_pose_theta_rad']])
        observations = torch.tensor([self.dataframe.loc[input_index, 'ft_wrench_xforce_N'], 
                                     self.dataframe.loc[input_index, 'ft_wrench_yforce_N'], 
                                     self.dataframe.loc[input_index, 'ft_wrench_ztorque_Nm']])
        target_states = torch.tensor([self.dataframe.loc[output_index, 'obj_pose_xcenter_m'],
                                      self.dataframe.loc[output_index, 'obj_pose_ycenter_m'], 
                                      self.dataframe.loc[output_index, 'obj_pose_theta_rad']])          
        return input_states, control_inputs, observations, target_states

class SequenceDataset(Dataset): 
    def __init__(self, path_list, mode, shift_length=1, sampling_frequency=-1, dataset_min=None, dataset_max=None):
        self.data = assemble_datasets(
            path_list=path_list, 
            mode=mode, 
            sampling_frequency=sampling_frequency, 
            shift_length=shift_length
        )
        self.shift_length = shift_length

        individual_dataset_lengths = []
        for dataset in self.data.datasets:
            individual_dataset_lengths.append(len(dataset))
        self.batch_indices = individual_dataset_lengths

        # self.dataset_mean = dataset_mean
        # self.dataset_std = dataset_std

        self.dataset_min = dataset_min
        self.dataset_max = dataset_max

    def __len__(self):
        return len(self.batch_indices) - 1

    def __getitem__(self, index):
        # X = torch.tensor(self.data.datasets[index].dataframe.values, dtype=torch.float32)
        # y = torch.tensor(self.data.datasets[index].dataframe.iloc[:,7:10].shift(-self.shift_length).values, dtype=torch.float32)

        # if not torch.is_tensor(self.dataset_mean):
        #     self.dataset_mean = torch.zeros(size=(1,12))

        # if not torch.is_tensor(self.dataset_std):
        #     self.dataset_std = torch.ones(size=(1,12))

        target_states = torch.tensor(self.data.datasets[index].dataframe.loc[:, 
                ['obj_pose_xcenter_m', 'obj_pose_ycenter_m', 'obj_pose_theta_rad']
            ].shift(-self.shift_length).values, dtype=torch.float32)
        nan_mask = ~torch.any(target_states.isnan(), dim=1)
        target_states = target_states[nan_mask]
        # target_states = (target_states-self.dataset_mean[:,9:12]) / self.dataset_std[:,9:12]
        if (self.dataset_min is not None) and (self.dataset_max is not None): 
            target_states = (target_states - self.dataset_min[:,9:12]) / (self.dataset_max[:,9:12] - self.dataset_min[:,9:12])

        input_states = torch.tensor(self.data.datasets[index].dataframe.loc[:, 
                ['obj_pose_xcenter_m', 'obj_pose_ycenter_m', 'obj_pose_theta_rad']
            ].values, dtype=torch.float32)
        input_states = input_states[nan_mask]
        # input_states = (input_states - self.dataset_mean[:,0:3]) / self.dataset_std[:,0:3]
        if (self.dataset_min is not None) and (self.dataset_max is not None): 
            input_states = (input_states - self.dataset_min[:,0:3]) / (self.dataset_max[:,0:3] - self.dataset_min[:,0:3])

        control_inputs = torch.tensor(self.data.datasets[index].dataframe.loc[:, 
                ['delta_obj_pose_xcenter_m', 'delta_obj_pose_ycenter_m', 'delta_obj_pose_theta_rad']
            ].values, dtype=torch.float32)
        control_inputs = control_inputs[nan_mask]
        # control_inputs = (control_inputs - self.dataset_mean[:,3:6]) / self.dataset_std[:,3:6]
        if (self.dataset_min is not None) and (self.dataset_max is not None): 
            control_inputs = (control_inputs - self.dataset_min[:,3:6]) / (self.dataset_max[:,3:6] - self.dataset_min[:,3:6])

        observations = torch.tensor(self.data.datasets[index].dataframe.loc[:, 
                ['ft_wrench_xforce_N', 'ft_wrench_yforce_N', 'ft_wrench_ztorque_Nm']
            ].values, dtype=torch.float32)
        observations = observations[nan_mask]
        # observations = (observations - self.dataset_mean[:,6:9]) / self.dataset_std[:,6:9]
        if (self.dataset_min is not None) and (self.dataset_max is not None): 
            observations = (observations - self.dataset_min[:,6:9]) / (self.dataset_max[:,6:9] - self.dataset_min[:,6:9])

        return input_states, control_inputs, observations, target_states