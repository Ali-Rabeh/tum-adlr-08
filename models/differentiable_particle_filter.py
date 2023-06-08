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

            if self.hparams['use_log_probs']: 
                self.weights[n,:] = -torch.log(self.hparams['num_particles'] * torch.ones(size=(self.hparams["num_particles"],1)))
            else:
                self.weights[n,:] = 1/self.hparams["num_particles"] * torch.ones(size=(self.hparams["num_particles"],1))

        assert self.particles.shape == (batch_size, self.hparams["num_particles"], 3)
        assert self.weights.shape == (batch_size, self.hparams["num_particles"], 1)

    def forward(self, states, control_inputs):
        """ Gets a differential update step for the states from the forward model and adds it to the particle states. 

        Args: 
            states (torch.tensor): 

            control_inputs (torch.tensor): 

        Returns: 
            self.particles (torch.tensor): Particles moved by one step according to the forward model.
        
        """
        delta_particles = self.forward_model.forward(states, control_inputs)
        propagated_particles = self.particles + delta_particles
        self.particles = propagated_particles
        assert self.particles.shape == (control_inputs.shape[0], self.hparams["num_particles"], 3)
        return propagated_particles

    def update(self, state, measurement):
        """ Passes the current measurements to the observation model, which returns their likelihoods given the particle states. 
            The weights are then updated with these likelihoods.

        Args: 
            state (torch.tensor): 

            measurement (torch.tensor):

        Returns:
        
        """
        likelihoods = self.observation_model(state, measurement)

        if self.hparams['use_log_probs']: 
            self.weights = self.weights + likelihoods
            self.weights = self.weights - torch.logsumexp(self.weights, dim=1, keepdim=True)
        else:
            self.weights = self.weights * likelihoods
            self.weights = self.weights / torch.sum(self.weights, dim=1, keepdim=True)

    def resample(self):
        """ 
        
        """
        return NotImplementedError

    def estimate(self): 
        """ Calculates the filter's state estimate as a weighted mean of its particles.

        """
        num_input_points = self.weights.shape[0]
        estimates = torch.zeros(size=(num_input_points,3))
        for n in range(num_input_points):
            weights_transposed = torch.t(self.weights[n,:,:])
            if self.hparams['use_log_probs']:
                estimates[n,:] = torch.matmul(torch.exp(weights_transposed), self.particles[n,:,:])
            else:
                estimates[n,:] = torch.matmul(weights_transposed, self.particles[n,:,:])

        assert estimates.shape == (num_input_points, 3)
        return estimates

    def step(self, control_input, measurement): 
        """ Performs one step of particle filter estimation. 

        Args: 
            control_input (torch.tensor): The current control inputs, for giving guidance to the forward model. 

            measurement (torch.tensor): The current measurements, to weight the propagated particles. 

        Returns: 
            estimate (torch.tensor): 

        """
        _ = self.forward(self.particles, control_input)
        self.update(self.particles, measurement)
        # self.resample()
        estimate = self.estimate()
        return estimate



