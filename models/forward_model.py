import torch
import torch.nn as nn
import numpy as np

# Largely based on: 
# https://github.com/zizo1111/tum-adlr-ws21-10/blob/main/pf/models/motion_model.py

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.1)

class ForwardModel(nn.Module):
    def __init__(self, input_dimension, state_dimension): 
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dimension, state_dimension),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(state_dimension, state_dimension)
        )
        self.model.apply(init_weights)

    def forward(self, particle_states: torch.Tensor):
        """ Returns delta_x, that is, a prediction for the small increment from last time step to the next: x_{t+1} = x_{t} + delta_x """

        batch_size = particle_states.shape[0]
        number_of_particles = particle_states.shape[1]
        # print(f"Batch size: {batch_size} | # Particles: {number_of_particles}")

        particle_states.to(self.device)

        # print(f"Particle states shape: {particle_states.shape}")
        out = self.model(particle_states) # do a forward pass 

        # TODO: get the mean and covariance for the predicted state here --> split the output of the model
        predicted_mean = out
        predicted_covariance = torch.eye(3)

        # Reparameterization trick apparently --> How does this work?
        # we need to use rsample instead of sample here to enable backpropagation (rsample: sampling using reparameterization trick)
        predicted_particle_states_diff = torch.distributions.MultivariateNormal(predicted_mean, predicted_covariance).rsample()

        return predicted_particle_states_diff
