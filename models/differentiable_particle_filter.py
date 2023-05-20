import torch
import numpy as np

from forward_model import ForwardModel

class DifferentiableParticleFilter:
    def __init__(self, number_of_particles, state_dimension):
        self.number_of_particles = number_of_particles
        self.particles = np.zeros((number_of_particles, state_dimension))
        self.weights = np.zeros((number_of_particles, state_dimension))

        self.initial_state = None

        self.forward_model = None
        self.measurement_model = None

        self.current_best_estimate_mean = None

    def initialize(self, initial_state, initial_covariance):
        """ Initializes the particles. """
        self.initial_state = initial_state
        self.particles = torch.distributions.MultivariateNormal(initial_state, initial_covariance).sample((self.particles.shape))
        # self.particles = self.initial_state + np.random.randn(self.particles.shape[0], self.particles.shape[1]) @ initial_covariance

    def forward(self, measurement):

        # 1. Apply forward model to predict the next particles' states

        # 2. do a resampling step if necessary 

        # 3. Apply observation model to get the likelihood for each particle

        # 4. Compute estimate as weighted average over the particles

        return NotImplementedError

    def proposal_update(self):
        """ Implements the one-step proposal function for the particles' next positions. """
        return NotImplementedError

    def measurement_update(self):

        return NotImplementedError

    def measurement_likelihood_function(self):
        """Calculates particle weights. """
        return NotImplementedError

    def resample(self): 
        return NotImplementedError

    def calculate_current_best_estimate(self):
        return NotImplementedError

    def update(self):
        return NotImplementedError


