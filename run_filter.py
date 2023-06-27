import os
import torch 
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.forward_model import ForwardModel
from models.observation_model import ObservationModel
from models.differentiable_particle_filter import DifferentiableParticleFilter

from util.data_file_paths import test_path_list
from util.dataset import SequenceDataset

hparams = {
    'mode': 'shifted',
    'shift_length': 1, 
    'sampling_frequency': 50, 
    'batch_size': 1, 

    'num_particles': 500, 
    'initial_covariance': torch.diag(torch.tensor([0.01, 0.01, 0.01]))
}

def visualize_particles(diff_particle_filter, filter_estimate, gt_pose):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(diff_particle_filter.particles[:,:,0].detach().numpy(), diff_particle_filter.particles[:,:,1].detach().numpy(), marker='+', label='Particles')
    ax.scatter(filter_estimate[:,0].detach().numpy(), filter_estimate[:,1].detach().numpy(), color='r', marker='+', label='Current estimate')
    ax.scatter(gt_pose[0], gt_pose[1], color='g', marker='o', label='Ground truth')
    ax.set_title("Current particles")
    ax.legend()
    ax.grid()
    return fig, ax

def visualize_weights(diff_particle_filter):
    fig, ax = plt.subplots(1, 1)
    weights = diff_particle_filter.weights
    weights = weights.squeeze().detach().numpy()
    ax.hist(weights)
    ax.set_title("Current weights")
    ax.grid()
    return fig, ax

def main(): 
    # 1. assemble the test dataset
    test_dataset = SequenceDataset(
        test_path_list, 
        mode=hparams['mode'], 
        sampling_frequency = hparams["sampling_frequency"],
        shift_length=hparams["shift_length"]
    )
    test_dataloader = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False)

    # 2. load the trained filter
    dpf = torch.load("models/saved_models/20230627_PretrainEpochs_20_3_Epochs_50_3_SequenceLengths_1_2_4.pth")

    # 3. for each sequence in the test dataset do: 
    for batch, (input_states, control_inputs, observations, target_states) in enumerate(test_dataloader):
        # sequence_length = X.shape[1]
        sequence_length = 10

        # states = X[:,7:10] # object states
        # control_inputs = X[:,4:7] # force / torque
        # measurements = X[:,1:7] # tip pose + force/torque
        print(input_states.shape)

        dpf_estimates = torch.zeros(size=(sequence_length, 3))

        # 3.1 initialize the filter
        initial_input_size = 1
        initial_state = input_states[:,0,:].unsqueeze(dim=1)
        dpf.initialize(1, initial_state, hparams['initial_covariance'])
        dpf_estimates[0,:] = dpf.estimate()
        print(f"Initial estimate: {dpf_estimates[0,:]}")

        # 3.2 step the filter through the sequence
        for n in range(1, sequence_length):
            current_control_inputs = control_inputs[:,n,:]
            current_measurements = observations[:,n,:]

            estimate = dpf.step(current_control_inputs, current_measurements)
            print(f"Step: {n} | Ground truth: {target_states[:,n,:]} | Filter estimate: {estimate}")
            dpf_estimates[n,:] = estimate

            # _, _ = visualize_particles(dpf, estimate, y[n,:])
            # _, _ = visualize_weights(dpf)
            # plt.show()

        dpf_estimates = dpf_estimates.detach().numpy()

        # 3.3 visualize the estimated state against the ground truth
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
        fig = plt.figure(num=batch)
        plt.suptitle("Test Sequence "+str(batch+1))
        ax1 = plt.subplot(gs[0,:])
        ax2 = plt.subplot(gs[1,0])
        ax3 = plt.subplot(gs[1,1])
        ax4 = plt.subplot(gs[1,2])

        ax1.scatter(target_states[:,:10,0], target_states[:,:10,1], label="Ground Truth")
        ax1.scatter(dpf_estimates[:10,0], dpf_estimates[:10,1], marker='+', label='Filter Estimate')
        ax1.axis('equal')
        ax1.set_title("Position")
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.legend()
        ax1.grid()

        ax2.plot(range(sequence_length), target_states[:,:10,0].squeeze())
        ax2.plot(range(sequence_length), dpf_estimates[:10,0])
        ax2.set_title("X-Position")
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('x_pos (m)')
        ax2.grid()

        ax3.plot(range(sequence_length), target_states[:,:10,1].squeeze())
        ax3.plot(range(sequence_length), dpf_estimates[:10,1])
        ax3.set_title("Y-Position")
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('y_pos (m)')
        ax3.grid()

        ax4.plot(range(sequence_length), target_states[:,:10,2].squeeze())
        ax4.plot(range(sequence_length), dpf_estimates[:10,2])
        ax4.set_title("Orientation")
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('theta (rad)')
        ax4.grid()

        plt.show()

    print("Done.")

if __name__ == '__main__':
    main()
