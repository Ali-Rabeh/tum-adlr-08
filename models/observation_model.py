import torch 
import torch.nn as nn
import numpy as np

class ObservationModel(nn.Module):
    def __init__(self, use_log_probs):
        """This will output weights for the particles.
        
        """
        super().__init__()

        if use_log_probs:
            last_layer = nn.Identity()
        else:
            last_layer = nn.Sigmoid()

        self.model = nn.Sequential(
            nn.Linear(7, 18), 
            nn.ReLU(),
            nn.Linear(18, 18), 
            nn.ReLU(),
            nn.Linear(18, 18),
            nn.ReLU(),
            nn.Linear(18, 1),
            last_layer
        )

    def forward(self, particle_states, measurements):
        """Calculates the likelihoods of current measurements given the particle states. 

        Args: 
            particle_states (torch.tensor): A tensor with size (batch_size, num_particles, 4) representing the particle states 
                                            in the continuous representation.
            measurements (torch.tensor): The current measured forces/torques.
        """
        num_input_points = measurements.shape[0]
        num_particles = particle_states.shape[1]

        state_dim = particle_states.shape[2]
        measurements_dim = measurements.shape[1]

        measurements_reshaped = torch.zeros(size=(num_input_points, num_particles, measurements_dim))
        for n in range(num_input_points):
            measurements_reshaped[n,:,:] = measurements[n,:].repeat(num_particles,1)
        assert measurements_reshaped.shape == (num_input_points, num_particles, measurements_dim)

        network_inputs = torch.concat((particle_states, measurements_reshaped), dim=2)
        assert network_inputs.shape == (num_input_points, num_particles, state_dim+measurements_dim)

        likelihoods = self.model(network_inputs)
        assert likelihoods.shape == (num_input_points, num_particles, 1)

        return likelihoods