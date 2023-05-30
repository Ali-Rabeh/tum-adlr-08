import os
import torch 
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from models.forward_model import ForwardModel
from models.observation_model import ObservationModel
from models.differentiable_particle_filter import DifferentiableParticleFilter

from util.data_file_paths import train_path_list
from util.dataset import assemble_datasets

hparams = {
    'sequence_length': 2, 
    'sampling_frequency': 50, 
    'batch_size': 1, 

    'num_particles': 500, 
    'initial_covariance': torch.diag(torch.tensor([0.1, 0.1, 0.01]))
}

def visualize_particles(diff_particle_filter, filter_estimate, gt_pose):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(diff_particle_filter.particles[:,0].detach().numpy(), diff_particle_filter.particles[:,1].detach().numpy(), marker='+', label='Particles')
    ax.scatter(filter_estimate[:,0].detach().numpy(), filter_estimate[:,1].detach().numpy(), color='r', marker='+', label='Current estimate')
    ax.scatter(gt_pose[0], gt_pose[1], color='g', marker='o', label='Ground truth')
    ax.set_title("Current particles")
    ax.legend()
    ax.grid()
    return fig, ax

def visualize_weights(diff_particle_filter):
    fig, ax = plt.subplots(1, 1)
    ax.hist(diff_particle_filter.weights.detach().numpy())
    ax.set_title("Current weights")
    ax.grid()
    return fig, ax

def main(): 
    train_dataset = assemble_datasets(
        train_path_list, 
        mode='shifted', 
        sampling_frequency=hparams['sampling_frequency'], 
        sequence_length=hparams['sequence_length'])

    train_dataset_length = []
    print(f"Number of individual datasets: {len(train_dataset.datasets)}")
    for dataset in train_dataset.datasets:
        train_dataset_length.append(len(dataset))
    print(train_dataset_length)

    train_dataloader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=False, drop_last=True)
    print(len(train_dataloader))

    forward_model = ForwardModel()
    observation_model = ObservationModel()
    dpf = DifferentiableParticleFilter(hparams, forward_model, observation_model)

    initial_X, initial_y = train_dataloader.dataset.__getitem__(0)
    initial_state = initial_X[7:10]
    dpf.initialize(initial_state, hparams["initial_covariance"])

    control_input = initial_X[4:7]
    measurement = initial_X[1:7]
    estimate = dpf.step(dpf.particles, control_input, measurement)

    gt_pose = initial_y
    fig_particles, ax_particles = visualize_particles(dpf, estimate, gt_pose)
    fig_weights, ax_weights = visualize_weights(dpf)
    plt.show()

if __name__ == '__main__':
    main()
