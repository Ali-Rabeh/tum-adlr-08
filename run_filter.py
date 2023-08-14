import os
import torch 
from torch.utils.data import DataLoader

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from models.forward_model import ForwardModel
from models.differentiable_particle_filter import DifferentiableParticleFilter

from util.data_file_paths import test_path_list, test_mean, test_std, test_min, test_max
from util.dataset import SequenceDataset

from camera_helpers import ImageGenerator

hparams = {
    'mode': 'shifted',
    'shift_length': 1, 
    'sampling_frequency': 50, 
    'batch_size': 1, 

    'use_images_for_forward_model': False,
    'use_forces_for_observation_model': False,
    'use_images_for_observation_model': True,
    'model_path': "models/saved_models/final/20230807_DPF_PretrainedForwardModel_ImageObservations.pth",

    'num_particles': 1000, 
    'initial_covariance': torch.diag(torch.tensor([0.2, 0.2, 0.2])), 

    'use_log_probs': True,
    'use_resampling': True,
    'resampling_soft_alpha': 0.01,

    'record_animations': True
}

sns.set_theme()
image_generator = ImageGenerator()
mse_loss = torch.nn.MSELoss()

def visualize_particles(particles, filter_estimate, gt_pose, previous_estimates=None, previous_gt_poses=None):
    """ Scatters the current particles along with the resulting filter estimate and the current ground truth. 
        If desired also scatters the preceding estimates and ground truth poses. 

    Args: 
        particles (torch.tensor): A tensor of size (batch_size x num_particles x 3) holding the poses of the current particles. 
        filter_estimate (torch.tensor): A tensor of size (1 x 3) representing the current filter estimate. 
        gt_pose (toch.tensor): A tensor of size (1 x 3) representing the current ground truth pose. 
        previous_estimates (torch.tensor): A tensor of size (previous_timesteps x 3) representing the previous filter estimates. 
        previous_gt_poses (torch.tensor): A tensor of size (previous_timesteps x 3) representing the previous ground truth poses. 

    Returns; 
        fig (plt.figure): Handle to the current matplotlib figure used for drawing
        ax (plt.axis): Handle to the current matplot axis used for drawing
    
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
    # ax.grid()
    plt.tight_layout()
    return fig, ax

def visualize_weights(diff_particle_filter):
    """ Plots a histogram of the current particle weight distribution. 

    Args: 
        diff_particle_filter (DifferentiableParticleFilter): The filter whose weights are to be plotted. 

    Returns: 
        fig (plt.figure): Handle to the current matplotlib figure used for drawing. 
        ax (plt.axis): Handle to the current matplot axis used for drawing.         

    """
    fig, ax = plt.subplots(1, 1)
    weights = diff_particle_filter.weights
    weights = weights.squeeze().detach().numpy()
    ax.hist(weights)
    ax.set_title("Current weights")
    ax.grid()
    return fig, ax

def convertFigureToImage(fig):
    """ Converts a plt.figure object to a cv2 image. Used for the animations.
    
    Args: 
        fig (plt.figure):  Handle to the figure that is to be converted. 

    Returns: 
        image (np.array): The converted figure in a format usable by opencv. 

    """
    fig.canvas.draw()
    image = np.array(fig.canvas.get_renderer()._renderer)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert from RGB to cv2's default BGR
    return image

def unnormalize_min_max(x, min, max): 
    """ Reverses the normalization of x. 
    
    Args: 
        x (torch.tensor): Tensor that is to be unnormalized. 
        min (torch.tensor): Minimum of x before it was normalized, has to be the same shape as x. 
        max (torch.tensor): Maximum of x before it was normalized, has to be the same shape as x. 

    Returns: 
        out (torch.tensor): Unnormalized version of x. 

    """
    out = (max-min) * x + min
    assert out.shape == x.shape
    return out

def unnormalize_mean_std(x, mean, std):
    """ Reverses the standardization of x. 
    
    Args: 
        x (torch.tensor): Tensor whose standardization you want to reverse. 
        mean (torch.tensor): Mean of x before it was standardized, has to be the same shape as x. 
        std (torch.tensor): Standard deviation of x before it was standardized, has to be the same shape as x.

    Returns; 
        out (torch.tensor): Input x with the same mean and standard deviation as before the standardization. 

    """
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
    dpf = torch.load(hparams['model_path'])
    print(dpf)

    # 3. for each sequence in the test dataset do: 
    rmse_sequences_states = torch.zeros(size=(len(test_dataloader), 3))
    rmse_sequences_position = torch.zeros(size=(len(test_dataloader), 1))
    for batch, (input_states, control_inputs, observations, target_states) in enumerate(test_dataloader):
        sequence_length = input_states.shape[1]

        dpf_estimates = torch.zeros(size=(sequence_length, 3))

        # 3.1 initialize the filter
        initial_input_size = 1
        initial_state = input_states[:,0,:].unsqueeze(dim=1)

        dpf.initialize(1, initial_state, hparams['initial_covariance'])
        dpf_estimates[0,:] = dpf.estimate()
        # print(f"Initial estimate: {dpf_estimates[0,:]}")

        # start the animation recording
        if hparams['record_animations']:
            video_path = "experiments/animations/sequence" + str(batch) + "_test.avi"
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps=2, frameSize=(640, 480))

            fig, _ = visualize_particles(
                unnormalize_min_max(dpf.particles, min=test_min[:,9:12].unsqueeze(dim=0), max=test_max[:,9:12].unsqueeze(dim=0)), 
                unnormalize_min_max(dpf_estimates[0,:].unsqueeze(dim=0), min=test_min[:,9:12], max=test_max[:,9:12]), 
                unnormalize_min_max(target_states[:,0,:].squeeze(), min=test_min[:,9:12].squeeze(), max=test_max[:,9:12].squeeze())
            )
            video_writer.write(convertFigureToImage(fig))
            plt.close(fig)

        # 3.2 step the filter through the sequence
        for n in range(1, sequence_length):
            current_control_inputs = control_inputs[:,n,:].unsqueeze(dim=1)
            current_measurements = observations[:,n,:]
            
            # generate an image
            current_image = image_generator.generate_image(
            	unnormalize_min_max(input_states[:,n,:].squeeze(), test_min[:,0:3].squeeze(), test_max[:,0:3].squeeze())
            )
            current_image = torch.tensor(current_image[None, None, :, :], dtype=torch.float32) / 255.0

            #  filter estimate for the current time step depending on the used observation model
            if not hparams['use_images_for_forward_model'] and \
                   hparams['use_forces_for_observation_model'] and \
               not hparams['use_images_for_observation_model']:
                estimate = dpf.step(current_control_inputs, measurement=current_measurements)

            if not hparams['use_images_for_forward_model'] and \
               not hparams['use_forces_for_observation_model'] and \
                   hparams['use_images_for_observation_model']:
                estimate = dpf.step(current_control_inputs, image=current_image)

            if hparams['use_images_for_forward_model'] or \
              (hparams['use_forces_for_observation_model'] and hparams['use_images_for_observation_model']): 
                estimate = dpf.step(current_control_inputs, measurement=current_measurements, image=current_image)
            
            dpf_estimates[n,:] = estimate
            # print(f"Step: {n} | Ground truth: {target_states[:,n,:]} | Filter estimate: {estimate}")

            if hparams['record_animations']:
                fig, _ = visualize_particles(
                    unnormalize_min_max(dpf.particles, min=test_min[:,9:12].unsqueeze(dim=0), max=test_max[:,9:12].unsqueeze(dim=0)), 
                    unnormalize_min_max(estimate, min=test_min[:,9:12], max=test_max[:,9:12]), 
                    unnormalize_min_max(target_states[:,n,:].squeeze(), min=test_min[:,9:12].squeeze(), max=test_max[:,9:12].squeeze()),
                    previous_estimates=unnormalize_min_max(dpf_estimates[0:n,:], min=test_min[:,9:12].squeeze(), max=test_max[:,9:12].squeeze()),
                    previous_gt_poses=unnormalize_min_max(target_states[:,0:n,:].squeeze(), min=test_min[:,9:12].squeeze(), max=test_max[:,9:12].squeeze())
                )
                video_writer.write(convertFigureToImage(fig))
                plt.close(fig)

            if (n > 20) and torch.norm(target_states[:,n+1,:] - target_states[:,n,:], p=2) < 1e-4: 
                sequence_length = n+1
                break

        # destroy everything needed for the animations
        if hparams['record_animations']:
            video_writer.release()
            cv2.destroyAllWindows()
            plt.close('all')

        dpf_estimates = dpf_estimates.detach().numpy()
        dpf_estimates = dpf_estimates[0:n+1,:]
        target_states = target_states[:,0:n+1,:]

        # unnormalize for better plotting
        dpf_estimates = unnormalize_min_max(dpf_estimates, min=test_min[:,9:12], max=test_max[:,9:12])
        target_states = unnormalize_min_max(target_states, min=test_min[:,9:12], max=test_max[:,9:12])
        # print(f"Target States unnormalized: {target_states}")

        # calculate error metrics
        target_states_squeezed = torch.squeeze(target_states)
        rmse_sequences_states[batch,0] = torch.sqrt(mse_loss(dpf_estimates[:,0], target_states_squeezed[:,0]))
        rmse_sequences_states[batch,1] = torch.sqrt(mse_loss(dpf_estimates[:,1], target_states_squeezed[:,1]))
        rmse_sequences_states[batch,2] = torch.sqrt(mse_loss(dpf_estimates[:,2], target_states_squeezed[:,2]))
        rmse_sequences_position[batch,0] = torch.sqrt(mse_loss(dpf_estimates[:,0:2], target_states_squeezed[:,0:2]))

        # 3.3 visualize the estimated state against the ground truth
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
        fig = plt.figure(num=batch, figsize=(8, 6), dpi=80)
        plt.suptitle("Test Sequence "+str(batch+1))

        ax1 = fig.add_subplot(gs[0,:])
        ax1.scatter(target_states[:,0:sequence_length,0], target_states[:,0:sequence_length,1], label="Ground Truth")
        ax1.scatter(dpf_estimates[0:sequence_length,0], dpf_estimates[0:sequence_length,1], marker='+', label='Filter Estimate')
        ax1.axis('equal')
        ax1.set_title("Position")
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.legend()
        # ax1.grid()

        ax2 = fig.add_subplot(gs[1,0])
        ax2.plot(range(sequence_length), target_states[:,0:sequence_length,0].squeeze())
        ax2.plot(range(sequence_length), dpf_estimates[0:sequence_length,0])
        ax2.set_title("X-Position")
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('x_pos (m)')
        # ax2.grid()

        ax3 = fig.add_subplot(gs[1,1])
        ax3.plot(range(sequence_length), target_states[:,0:sequence_length,1].squeeze())
        ax3.plot(range(sequence_length), dpf_estimates[0:sequence_length,1])
        ax3.set_title("Y-Position")
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('y_pos (m)')
        # ax3.grid()

        ax4 = fig.add_subplot(gs[1,2])
        ax4.plot(range(sequence_length), target_states[:,0:sequence_length,2].squeeze())
        ax4.plot(range(sequence_length), dpf_estimates[0:sequence_length,2])
        ax4.set_title("Orientation")
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('theta (rad)')
        # ax4.grid()

        gs.tight_layout(fig)

        # plt.savefig("experiments/figures/sequence"+str(batch)+".png", format='png')
        plt.show()

    print("Done.")
    print(f"RMSE component-wise: {rmse_sequences_states}")
    print(f"RMSE component-wise whole test set: {torch.sqrt(torch.mean(rmse_sequences_states**2, dim=0))}")
    print(f"RMSE position-wise: {rmse_sequences_position}")
    print(f"RMSE position-wise whole test set: {torch.sqrt(torch.mean(rmse_sequences_position**2, dim=0))}")

if __name__ == '__main__':
    main()
