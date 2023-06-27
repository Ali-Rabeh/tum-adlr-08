import torch
import torch.nn as nn
import numpy as np

# Largely based on: 
# https://github.com/zizo1111/tum-adlr-ws21-10/blob/main/pf/models/motion_model.py

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)

class ForwardModel(nn.Module):
    def __init__(self): 
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = nn.Sequential(
            nn.Linear(8, 12),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(12, 12),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(12, 4)
        )

        self.model.apply(init_weights)

    def forward(self, particle_states, control_inputs):
        """ Returns delta_x, that is, a prediction for the small increment from last time step to the next: x_{t+1} = x_{t} + delta_x 
        
        Args: 
            particle_states (torch.tensor): dimensions = (batch_size, num_particles, state_dim), given as continuous representation
            control_inputs (torch.tensor): dimensions = (batch_size, control_dim), given as continuous representation
        """

        # get the relevant dimensions
        num_input_points = particle_states.shape[0]
        num_particles = particle_states.shape[1]
        state_dim = particle_states.shape[2]

        control_dim = control_inputs.shape[2]
        assert control_inputs.shape == (1, 1, control_dim)

        # reshape the control input
        control_inputs_reshaped = torch.zeros(size=(num_input_points, num_particles, control_dim))
        for n in range(num_input_points):
            control_inputs_reshaped[n,:,:] = control_inputs[0,n,:].repeat(num_particles,1)
        assert control_inputs_reshaped.shape == (num_input_points, num_particles, control_dim)

        # concatenate the inputs to the network together
        network_inputs = torch.concat((particle_states, control_inputs_reshaped), dim=2)
        assert network_inputs.shape == (num_input_points, num_particles, state_dim+control_dim)
        # print(network_inputs.shape)

        # do a forward pass
        network_inputs.to(self.device)
        out = self.model(network_inputs)

        # sample prediction
        predicted_mean = out
        predicted_covariance = 0.1*torch.eye(4)

        # Reparameterization trick apparently --> How does this work?
        # we need to use rsample instead of sample here to enable backpropagation (rsample: sampling using reparameterization trick)
        predicted_particle_states_diff = torch.distributions.MultivariateNormal(predicted_mean, predicted_covariance).rsample()

        return predicted_particle_states_diff
