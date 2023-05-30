import os
import sys
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, ConcatDataset

from dataset import NormalDataset, ShiftedDataset, assemble_datasets
from data_file_paths import train_path_list, validation_path_list, test_path_list

def plot_sequence_variables(sequence_data):
    time_vector = sequence_data.dataframe["time_sec"] - sequence_data.dataframe["time_sec"][0]

    fig, axes = plt.subplots(3, 3)

    axes[0, 0].plot(time_vector, sequence_data.dataframe.loc[:,'tip_pose_xpos_m'])
    axes[0, 0].set_title("tip_pose_xpos_m")
    axes[0, 0].grid()

    axes[0, 1].plot(time_vector, sequence_data.dataframe.loc[:,'tip_pose_ypos_m'])
    axes[0, 1].set_title("tip_pose_ypos_m")
    axes[0, 1].grid()

    axes[0, 2].plot(time_vector, sequence_data.dataframe.loc[:,"tip_pose_theta_pos_rad"])
    axes[0, 2].set_title("tip_pose_theta_pos_rad")
    axes[0, 2].grid()

    axes[1, 0].plot(time_vector, sequence_data.dataframe.loc[:,"obj_pose_xcenter_m"])
    axes[1, 0].set_title("obj_pose_xcenter_m")
    axes[1, 0].grid()

    axes[1, 1].plot(time_vector, sequence_data.dataframe.loc[:,"obj_pose_ycenter_m"])
    axes[1, 1].set_title("obj_pose_ycenter_m")
    axes[1, 1].grid()

    axes[1, 2].plot(time_vector, sequence_data.dataframe.loc[:,"obj_pose_theta_rad"])
    axes[1, 2].set_title("obj_pose_theta_rad")
    axes[1, 2].grid()

    axes[2, 0].plot(time_vector, sequence_data.dataframe.loc[:,"ft_wrench_xforce_N"])
    axes[2, 0].set_title("ft_wrench_xforce_N")
    axes[2, 0].grid()

    axes[2, 1].plot(time_vector, sequence_data.dataframe.loc[:,"ft_wrench_yforce_N"])
    axes[2, 1].set_title("ft_wrench_yforce_N")
    axes[2, 1].grid()

    axes[2, 2].plot(time_vector, sequence_data.dataframe.loc[:,"ft_wrench_ztorque_Nm"])
    axes[2, 2].set_title("ft_wrench_ztorque_Nm")
    axes[2, 2].grid()

    return time_vector, fig, axes

def plot_object_and_tip_pose(sequence_data):
    time_vector = sequence_data.dataframe["time_sec"] - sequence_data.dataframe["time_sec"][0]

    fig, axes = plt.subplots(1, 2)

    axes[0].scatter(sequence_data.dataframe.loc[:,'tip_pose_xpos_m'], sequence_data.dataframe.loc[:,'tip_pose_ypos_m'])
    axes[0].set_title("Tip position")
    axes[0].grid()

    axes[1].scatter(sequence_data.dataframe.loc[:,'obj_pose_xcenter_m'], sequence_data.dataframe.loc[:,'obj_pose_ycenter_m'])
    axes[1].set_title("Object center position")
    axes[1].grid()

    return time_vector, fig, axes

# Test for the single source sequence dataset
sequence_length = 2
sampling_frequency = 50
batch_size = 16

train_sequences = assemble_datasets(train_path_list, mode="shifted", sampling_frequency=sampling_frequency, sequence_length=sequence_length)
validation_sequences = assemble_datasets(validation_path_list, mode="shifted", sampling_frequency=sampling_frequency, sequence_length=sequence_length)
test_sequences = assemble_datasets(test_path_list, mode="shifted", sampling_frequency=sampling_frequency, sequence_length=sequence_length)

#######################
# Check the train set # 
#######################
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[0])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[1])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[2])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[3])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[4])

# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[5])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[6])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[7])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[8])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[9])

# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[10])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[11])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[12])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[13])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[14])

# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[15])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[16])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[17])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[18])
# time_vector, _, _ = plot_sequence_variables(train_sequences.datasets[19])
# plt.show()

############################
# Check the validation set #
############################
# time_vector, _, _ = plot_sequence_variables(validation_sequences.datasets[0])
# time_vector, _, _ = plot_sequence_variables(validation_sequences.datasets[1])
# time_vector, _, _ = plot_sequence_variables(validation_sequences.datasets[2])

# time_vector, _, _ = plot_sequence_variables(validation_sequences.datasets[3])
# time_vector, _, _ = plot_sequence_variables(validation_sequences.datasets[4])
# time_vector, _, _ = plot_sequence_variables(validation_sequences.datasets[5])

# time_vector, _, _ = plot_sequence_variables(validation_sequences.datasets[6])
# time_vector, _, _ = plot_sequence_variables(validation_sequences.datasets[7])
# time_vector, _, _ = plot_sequence_variables(validation_sequences.datasets[8])

# time_vector, _, _ = plot_sequence_variables(validation_sequences.datasets[9])
# time_vector, _, _ = plot_sequence_variables(validation_sequences.datasets[10])
# time_vector, _, _ = plot_sequence_variables(validation_sequences.datasets[11])
# plt.show()

######################
# Check the test set #
######################
time_vector, _, _ = plot_sequence_variables(test_sequences.datasets[0])
time_vector, _, _ = plot_sequence_variables(test_sequences.datasets[1])
time_vector, _, _ = plot_sequence_variables(test_sequences.datasets[2])

time_vector, _, _ = plot_sequence_variables(test_sequences.datasets[3])
time_vector, _, _ = plot_sequence_variables(test_sequences.datasets[4])
time_vector, _, _ = plot_sequence_variables(test_sequences.datasets[5])

time_vector, _, _ = plot_sequence_variables(test_sequences.datasets[6])
time_vector, _, _ = plot_sequence_variables(test_sequences.datasets[7])
time_vector, _, _ = plot_sequence_variables(test_sequences.datasets[8])

time_vector, _, _ = plot_sequence_variables(test_sequences.datasets[9])
time_vector, _, _ = plot_sequence_variables(test_sequences.datasets[10])
time_vector, _, _ = plot_sequence_variables(test_sequences.datasets[11])
plt.show()



# dataloader_sequences_single_source = DataLoader(sequences, batch_size=batch_size, shuffle=False)
# X_batch, y_batch = next(iter(dataloader_sequences_single_source))
# print(X_batch)
# print(y_batch)
# print(type(X_batch))
# print(type(y_batch))