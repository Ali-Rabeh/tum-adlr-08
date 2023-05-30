import torch
import torch.nn as nn
import numpy as np

def init_weights(module):
    """ 
    
    """
    if isinstance(module, (nn.Linear)):
        nn.init.kaiming_normal_(module.weight)

class DifferentiableParticleFilter(nn.Module):
    def __init__(self, hparams, forward_model, observation_model):
        super().__init__()
        
        self.hparams = hparams
        self.forward_model = forward_model
        self.observation_model = observation_model

        self.observation_model.apply(init_weights)

        self.particles = torch.zeros(size=[])
        self.weights = torch.zeros(size=[])

        self.current_best_estimate = torch.zeros(size=(1,3))

    def initialize(self, batch_size, initial_states, initial_covariance):
        """

        """
        self.particles = torch.zeros(size=(batch_size, self.hparams["num_particles"], 3))
        self.weights = torch.zeros(size=(batch_size, self.hparams["num_particles"], 1))
        for n in range(batch_size):
            self.particles[n,:,:] = initial_states[n,:] + torch.randn(self.hparams["num_particles"], 3) @ initial_covariance
            self.weights[n,:] = 1/self.hparams["num_particles"] * torch.ones(size=(self.hparams["num_particles"],1))

        assert self.particles.shape == (batch_size, self.hparams["num_particles"], 3)
        assert self.weights.shape == (batch_size, self.hparams["num_particles"], 1)

    def forward(self, states, control_inputs):
        """ 
        
        """
        delta_particles = self.forward_model.forward(states, control_inputs)
        self.particles = self.particles + delta_particles
        assert self.particles.shape == (control_inputs.shape[0], self.hparams["num_particles"], 3)
        return self.particles

    def update(self, state, measurement):
        """ 
        
        """
        likelihoods = self.observation_model(state, measurement)

        self.weights = self.weights * likelihoods
        self.weights = self.weights / torch.sum(self.weights)

    def resample(self):
        """ 
        
        """
        return NotImplementedError

    def estimate(self): 
        """

        """
        num_input_points = self.weights.shape[0]
        estimates = torch.zeros(size=(num_input_points,3))
        for n in range(num_input_points):
            weights_transposed = torch.t(self.weights[n,:,:])
            estimates[n,:] = torch.matmul(weights_transposed, self.particles[n,:,:])

        assert estimates.shape == (num_input_points, 3)
        return estimates

    def step(self, state, control_input, measurement): 
        """

        """
        _ = self.forward(self.particles, control_input)
        self.update(self.particles, measurement)
        # self.resample()
        estimate = self.estimate()
        return estimate



