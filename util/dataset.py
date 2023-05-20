import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as pyplot

class CustomDataset(Dataset):
    def __init__(self, path):
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

        self.dataframe = pd.merge_asof(self.tip_pose_df, self.ft_wrench_df, on='time_sec', direction='nearest')
        self.dataframe = pd.merge_asof(self.dataframe, self.obj_pose_df, on='time_sec', direction='nearest')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx): 
        # print(self.dataframe.iloc[idx,:])
        # return self.dataframe.iloc[idx,:].to_numpy(dtype=float), self.dataframe.iloc[idx,7:10].to_numpy(dtype=float)
        return torch.tensor(self.dataframe.iloc[idx,:], dtype=torch.float32), torch.tensor(self.dataframe.iloc[idx,7:10], dtype=torch.float32)