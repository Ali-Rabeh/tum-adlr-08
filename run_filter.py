import os
import torch 
from torch.utils.data import DataLoader

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.forward_model import ForwardModel
from models.observation_model import ObservationModel
from models.differentiable_particle_filter import DifferentiableParticleFilter

from util.data_file_paths import test_path_list, test_mean, test_std, test_min, test_max
from util.dataset import SequenceDataset

from camera_helpers import ImageGenerator

hparams = {
    'mode': 'shifted',
    'shift_length': 1, 
    'sampling_frequency': 50, 
    'batch_size': 1, 

    'use_forces': True,
    'use_images': False,

    'num_particles': 500, 
    'initial_covariance': torch.diag(torch.tensor([0.2, 0.2, 0.2])), 

    'use_log_probs': True,
    'use_resampling': True,
    'resampling_soft_alpha': 0.05,

    'record_animations': True
}

image_generator = ImageGenerator()

def visualize_particles(particles, filter_estimate, gt_pose, previous_estimates=None, previous_gt_poses=None):
    """ 
    
    """
    if previous_gt_poses is not None: 
        previous_gt_poses = torch.atleast_2d(previous_gt_poses)
        # print(f"Previous GT poses shape: {previous_gt_poses.shape}")

    fig = plt.figure(figsize=(8, 6), dpi=80) # creates a 640 by 480 figure
    ax = fig.add_subplot(1, 1, 1, autoscale_on=True)

    # scatter all particles
    ax.scatter(particles[:,:,0].detach().numpy(), particles[:,:,1].detach().numpy(), marker='+', label='Particles')

    # scatter the current estimate as well as previous ones
    ax.scatter(filter_estimate[:,0].detach().numpy(), filter_estimate[:,1].detach().numpy(), color='r', marker='+', label='Current estimate')
    if previous_estimates is not None: 
        ax.scatter(previous_estimates[:,0].detach().numpy(), previous_estimates[:,1].detach().numpy(), color='r', marker='+', alpha=0.5)

    # scatter the current ground truth pose as well as previous ones
    ax.scatter(gt_pose[0], gt_pose[1], color='g', marker='o', label='Ground truth')
    if previous_gt_poses is not None: 
        ax.scatter(previous_gt_poses[:,0], previous_gt_poses[:,1], color='g', marker='o', alpha=0.5)

    ax.set_title("Current particles")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
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

def convertFigureToImage(fig):
    """ Converts a plt.figure object to a cv2 image. Used for the animations.
    
    fig (plt.figure):

    """
    fig.canvas.draw()
    image = np.array(fig.canvas.get_renderer()._renderer)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert from RGB to cv2's default BGR
    return image

def unnormalize_min_max(x, min, max): 
    """ Unnormalizes x, which is assumed to be a tensor. """
    out = (max-min) * x + min
    assert out.shape == x.shape
    return out

def unnormalize_mean_std(x, mean, std):
    """ Unnormalizes x, which is assumed to be a tensor. """
    out = std * x + mean 
    assert out.shape == x.shape
    return out

def main(): 
    # 1. assemble the test dataset
    test_dataset = SequenceDataset(
        test_path_list, 
        mode=hparams['mode'], 
        sampling_frequency = hparams["sampling_frequency"],
        shift_length=hparams["shift_length"],
        dataset_min=test_min,
        dataset_max=test_max
    )
    test_dataloader = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False)

    # 2. load the trained filter
    dpf = torch.load("models/saved_models/20230806_OnlyForceObservationsWithPretrainedForwardModel.pth")
    # print(dpf)

    # 3. for each sequence in the test dataset do: 
    for batch, (input_states, control_inputs, observations, target_states) in enumerate(test_dataloader):
        sequence_length = input_states.shape[1]
        # sequence_length = 20

        dpf_estimates = torch.zeros(size=(sequence_length, 3))

        # 3.1 initialize the filter
        initial_input_size = 1
        initial_state = input_states[:,0,:].unsqueeze(dim=1)

        dpf.initialize(1, initial_state, hparams['initial_covariance'])
        dpf_estimates[0,:] = dpf.estimate()
        print(f"Initial estimate: {dpf_estimates[0,:]}")

        # start the animation recording
        if hparams['record_animations']:
            video_path = "experiments/animations/sequence" + str(batch) + ".avi"
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps=1, frameSize=(640, 480))

            fig, _ = visualize_particles(
                unnormalize_min_max(dpf.particles, min=test_min[:,9:12].unsqueeze(dim=0), max=test_max[:,9:12].unsqueeze(dim=0)), 
                unnormalize_min_max(dpf_estimates[0,:].unsqueeze(dim=0), min=test_min[:,9:12], max=test_max[:,9:12]), 
                unnormalize_min_max(target_states[:,0,:].squeeze(), min=test_min[:,9:12].squeeze(), max=test_max[:,9:12].squeeze())
            )
            video_writer.write(convertFigureToImage(fig))

        # 3.2 step the filter through the sequence
        for n in range(1, sequence_length):
            current_control_inputs = control_inputs[:,n,:].unsqueeze(dim=1)
            current_measurements = observations[:,n,:]
            
            current_image = image_generator.generate_image(
            	unnormalize_min_max(input_states[:,n,:].squeeze(), test_min[:,0:3].squeeze(), test_max[:,0:3].squeeze())
            )
            current_image = torch.tensor(current_image[None, None, :, :], dtype=torch.float32) / 255.0

            if hparams['use_forces'] and not hparams['use_images']:
                estimate = dpf.step(current_control_inputs, measurement=current_measurements)
            if not hparams['use_forces'] and hparams['use_images']:
                estimate = dpf.step(current_control_inputs, image=current_image)
            
            dpf_estimates[n,:] = estimate
            print(f"Step: {n} | Ground truth: {target_states[:,n,:]} | Filter estimate: {estimate}")

            if hparams['record_animations']:
                fig, _ = visualize_particles(
                    unnormalize_min_max(dpf.particles, min=test_min[:,9:12].unsqueeze(dim=0), max=test_max[:,9:12].unsqueeze(dim=0)), 
                    unnormalize_min_max(estimate, min=test_min[:,9:12], max=test_max[:,9:12]), 
                    unnormalize_min_max(target_states[:,n,:].squeeze(), min=test_min[:,9:12].squeeze(), max=test_max[:,9:12].squeeze()),
                    previous_estimates=unnormalize_min_max(dpf_estimates[0:n,:], min=test_min[:,9:12].squeeze(), max=test_max[:,9:12].squeeze()),
                    previous_gt_poses=unnormalize_min_max(target_states[:,0:n,:].squeeze(), min=test_min[:,9:12].squeeze(), max=test_max[:,9:12].squeeze())
                )
                video_writer.write(convertFigureToImage(fig))

        # destroy everything needed for the animations
        if hparams['record_animations']:
            video_writer.release()
            cv2.destroyAllWindows()
            plt.close('all')

        dpf_estimates = dpf_estimates.detach().numpy()

        # unnormalize for better plotting
        dpf_estimates = unnormalize_min_max(dpf_estimates, min=test_min[:,9:12], max=test_max[:,9:12])
        target_states = unnormalize_min_max(target_states, min=test_min[:,9:12], max=test_max[:,9:12])
        print(f"Target States unnormalized: {target_states}")

        # 3.3 visualize the estimated state against the ground truth
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
        fig = plt.figure(num=batch)
        plt.suptitle("Test Sequence "+str(batch+1))
        ax1 = plt.subplot(gs[0,:])
        ax2 = plt.subplot(gs[1,0])
        ax3 = plt.subplot(gs[1,1])
        ax4 = plt.subplot(gs[1,2])

        ax1.scatter(target_states[:,0:sequence_length,0], target_states[:,0:sequence_length,1], label="Ground Truth")
        ax1.scatter(dpf_estimates[0:sequence_length,0], dpf_estimates[0:sequence_length,1], marker='+', label='Filter Estimate')
        ax1.axis('equal')
        ax1.set_title("Position")
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.legend()
        ax1.grid()

        ax2.plot(range(sequence_length), target_states[:,0:sequence_length,0].squeeze())
        ax2.plot(range(sequence_length), dpf_estimates[0:sequence_length,0])
        ax2.set_title("X-Position")
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('x_pos (m)')
        ax2.grid()

        ax3.plot(range(sequence_length), target_states[:,0:sequence_length,1].squeeze())
        ax3.plot(range(sequence_length), dpf_estimates[0:sequence_length,1])
        ax3.set_title("Y-Position")
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('y_pos (m)')
        ax3.grid()

        ax4.plot(range(sequence_length), target_states[:,0:sequence_length,2].squeeze())
        ax4.plot(range(sequence_length), dpf_estimates[0:sequence_length,2])
        ax4.set_title("Orientation")
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('theta (rad)')
        ax4.grid()

        plt.show()

    print("Done.")

if __name__ == '__main__':
    main()
