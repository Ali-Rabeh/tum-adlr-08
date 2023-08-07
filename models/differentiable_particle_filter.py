import torch
import torch.nn as nn
import numpy as np

from util.manifold_helpers import boxplus, radianToContinuous, continuousToRadian, weightedAngularMean

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

        self.particles = torch.zeros(size=[]) # particles are kept in a discontinuous representation!
        self.weights = torch.zeros(size=[])

        self.step_counter = 0

        self.current_best_estimate = torch.zeros(size=(1,3))

    def initialize(self, batch_size, initial_states, initial_covariance):
        """ Initializes the particle filter.

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

    def forward(self, states, control_inputs, image=None):
        """ Gets a differential update step for the states from the forward model and adds it to the particle states. 

        Args: 
            states (torch.tensor): 
            control_inputs (torch.tensor): 
            image (torch.tensor):

        Returns: 
            self.particles (torch.tensor): Particles moved by one step according to the forward model.
        """

        states = radianToContinuous(states)
        control_inputs = radianToContinuous(control_inputs)

        if image is None:
            delta_particles = self.forward_model.forward(states, control_inputs)
        else:
            delta_particles = self.forward_model.forward(states, control_inputs, image)

        # convert to continuous representation  
        delta_particles = continuousToRadian(delta_particles)

        propagated_particles = boxplus(self.particles, delta_particles)
        self.particles = propagated_particles
        assert self.particles.shape == (control_inputs.shape[0], self.hparams["num_particles"], 3)

        return propagated_particles

    def update(self, states, measurement=None, image=None):
        """ Passes the current measurements to the observation model, which returns their likelihoods given the particle states. 
            The weights are then updated with these likelihoods.

        Args: 
            states (torch.tensor): 
            measurement (torch.tensor):
            image (torch.tensor): 
        """

        states = radianToContinuous(states)

        if ((measurement is not None) and (image is None)) or self.hparams['use_images_for_forward_model']: # use only force measurements in the observation_model
            likelihoods = self.observation_model(states, measurement=measurement)

        if (measurement is None) and (image is not None) and not self.hparams['use_images_for_forward_model']: # use only images
            likelihoods = self.observation_model(states, image=image)

        if (measurement is not None) and (image is not None) and not self.hparams['use_images_for_forward_model']: # use both force measurements and images
            likelihoods = self.observation_model(states, measurement=measurement, image=image)

        if self.hparams['use_log_probs']: 
            self.weights = self.weights + likelihoods
            self.weights = self.weights - torch.logsumexp(self.weights, dim=1, keepdim=True) # normalize for valid probabilities
        else:
            self.weights = self.weights * likelihoods
            self.weights = self.weights / torch.sum(self.weights, dim=1, keepdim=True)

    def resample(self, soft_resample_alpha):
        """ Resample particles, currently only works for weights in log-space. 

        Args: 
            soft_resample_alpha (float):
        
        """
        soft_resample_alpha = torch.tensor(soft_resample_alpha, dtype=torch.float32)
        batch_size, num_particles, state_dim = self.particles.shape

        # calculate uniform weights
        uniform_log_weights = self.weights.new_full(
            (batch_size, num_particles, 1), float(-torch.log(torch.tensor(num_particles, dtype=torch.float32)))
        )

        if soft_resample_alpha < 1.0: # actual soft resampling
            sample_logits = torch.logsumexp(
                torch.stack([self.weights + torch.log(soft_resample_alpha), uniform_log_weights + torch.log(1.0 - soft_resample_alpha)], dim=0),
                dim=0
            )
            self.weights = self.weights - sample_logits
        else: # regular resampling, stops gradients
            sample_logits = self.weights
            self.weights = uniform_log_weights

        # draw indices according to the weights
        sample_logits = sample_logits.squeeze()
        distribution = torch.distributions.Categorical(logits=sample_logits)
        indices = distribution.sample(sample_shape=(num_particles,))

        # draw corresponding particles
        self.particles = self.particles[:, indices, :] + 0.05*torch.randn(size=self.particles.shape)
        assert self.particles.shape == (batch_size, num_particles, state_dim) 

    def estimate(self): 
        """ Calculates the filter's state estimate as a weighted mean of its particles.

        """
        num_input_points = self.weights.shape[0]
        estimates = torch.zeros(size=(num_input_points,3))
        for n in range(num_input_points):
            weights_transposed = torch.t(self.weights[n,:,:])
            if self.hparams['use_log_probs']:
                # print(torch.exp(weights_transposed))
                estimates[n,0:2] = torch.matmul(torch.exp(weights_transposed), self.particles[n,:,0:2])
                estimates[n,2] = weightedAngularMean(self.particles[n,:,2], torch.exp(weights_transposed))
            else:
                estimates[n,0:2] = torch.matmul(weights_transposed, self.particles[n,:,0:2])
                estimates[n,2] = weightedAngularMean(self.particles[n,:,2], weights_transposed)

        assert estimates.shape == (num_input_points, 3)
        return estimates

    def step(self, control_input, measurement=None, image=None): 
        """ Performs one step of particle filter estimation. 

        Args: 
            control_input (torch.tensor): The current control inputs, for giving guidance to the forward model. 
            measurement (torch.tensor): The current measurements, to weight the propagated particles. 

        Returns: 
            estimate (torch.tensor): 
        """

        # propagate particles forward
        if self.hparams['use_images_for_forward_model']: 
            _ = self.forward(self.particles, control_input, image=image)
        else: 
            _ = self.forward(self.particles, control_input)

        # incorporate observations
        self.update(self.particles, measurement=measurement, image=image)

        # get the filter's current estimate
        estimate = self.estimate()

        # currently, the resampling step is only written for weights in log-space
        if self.hparams['use_resampling'] and self.hparams['use_log_probs']: 
            if self.step_counter % 1 == 0: 
                self.resample(self.hparams['resampling_soft_alpha'])

        self.step_counter += 1

        return estimate



