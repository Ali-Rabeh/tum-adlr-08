import os
import sys

from torch.utils.data import DataLoader, ConcatDataset

from dataset import CustomDataset, SequenceDataset

path_list = [os.path.abspath("../pushdata/plywood/butter/motion_surface=plywood_shape=butter_a=0_v=10_i=0.000_s=0.000_t=0.000.json"), 
             os.path.abspath("../pushdata/plywood/butter/motion_surface=plywood_shape=butter_a=0_v=10_i=0.000_s=0.000_t=0.349.json"), 
             os.path.abspath("../pushdata/plywood/butter/motion_surface=plywood_shape=butter_a=0_v=10_i=0.000_s=0.000_t=0.698.json")]
             # os.path.abspath("../pushdata/plywood/butter/motion_surface=plywood_shape=butter_a=0_v=10_i=0.000_s=0.000_t=-0.349.json"), 
             # os.path.abspath("../pushdata/plywood/butter/motion_surface=plywood_shape=butter_a=0_v=10_i=0.000_s=0.000_t=-0.698.json")]

# # Test for the single source dataset
# trajectory = CustomDataset(path_list[0])
# print(trajectory.dataframe.head())

# dataloader_single_source = DataLoader(trajectory, batch_size=16, shuffle=False)
# batch = next(iter(dataloader_single_source))
# print(batch)
# print(type(batch))

# # Test for the multiple sources dataset
# list_of_datasets = []
# for path in path_list:
#     list_of_datasets.append(CustomDataset(path))
# trajectories = ConcatDataset(list_of_datasets)
# print(len(trajectories))

# dataloader_multiple_sources = DataLoader(trajectories, batch_size=64, shuffle=False)
# batch = next(iter(dataloader_multiple_sources))
# print(batch)
# print(type(batch))

# Test for the single source sequence dataset
sequence_length = 4
batch_size = 16
sequences = SequenceDataset(path_list[0], sequence_length)
print(len(sequences))

dataloader_sequences_single_source = DataLoader(sequences, batch_size=batch_size, shuffle=False)
X_batch, y_batch = next(iter(dataloader_sequences_single_source))
print(X_batch)
print(y_batch)
print(type(X_batch))
print(type(y_batch))