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

        self.particles = torch.zeros(size=[]) # particles are kept in a discontinuous representation!
        self.weights = torch.zeros(size=[])

        self.current_best_estimate = torch.zeros(size=(1,3))

    def boxplus(self, state, delta, scaling=1.0):
        """ Implements the boxplus operation for the SE(2) manifold. The implementation follows Lennart's manitorch code. 

        Args: 
            state (torch.tensor): A 1xnum_particlesx3 tensor representing the state [x, y, theta]
            delta (torch.tensor): A 1xnum_particlesx3 tensor representing the step towards the next state: [delta_x, delta_y, delta_theta]

        Returns:
            new_state (torch.tensor): A 1xnum_particlesx3 tensor representing the new state: [new_x, new_y, new_theta]. 
                                      Basically, new_state = state + delta, and we make sure that new_theta is between -pi and pi.
        """
        # print(f"State shape: {state.shape} | Delta shape: {delta.shape}")

        delta = scaling * delta
        new_state = state + delta
        new_state[:,:,2] = ((new_state[:,:,2] + np.pi) % (2*np.pi)) - np.pi
        return new_state

    def radianToContinuous(self, states):
        """ Maps the state [x, y, theta_rad] to the state [x, y, cos(theta_rad), sin(theta_rad)]. 
        
        Args: 
            states (torch.tensor): A tensor with size (batch_size, num_particles, 3) representing the particles states.

        Returns: 
            continuous_states (torch.tensor): A tensor with size (batch_size, num_particles, 4) representing a continuous version of the 
                                              particle states.
        """
        continuous_states = torch.zeros(size=(states.shape[0], states.shape[1], states.shape[2]+1))
        # print(continuous_states.shape)
        continuous_states[:,:,0:1] = states[:,:,0:1]
        continuous_states[:,:,2] = torch.cos(states[:,:,2])
        continuous_states[:,:,3] = torch.sin(states[:,:,2])
        return continuous_states

    def continuousToRadian(self, states):
        """ Maps states [x, y, cos(theta_rad), sin(theta_rad) to states [x, y, theta_rad]. 
        
        Args: 
            states (torch.tensor):

        Returns:
            discontinuous_states (torch.tensor):
        """

        discontinuous_states = torch.zeros(size=(states.shape[0], states.shape[1], states.shape[2]-1))
        discontinuous_states[:,:,0:1] = states[:,:,0:1]
        discontinuous_states[:,:,2] = torch.atan2(states[:,:,3], states[:,:,2])
        return discontinuous_states

    def weightedAngularMean(self, angles, weights):
        """Calculates the weighted angular mean. 

        Args: 

        Returns: 

        """
        return NotImplementedError

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

    def forward(self, states, control_inputs):
        """ Gets a differential update step for the states from the forward model and adds it to the particle states. 

        Args: 
            states (torch.tensor): 
            control_inputs (torch.tensor): 

        Returns: 
            self.particles (torch.tensor): Particles moved by one step according to the forward model.
        """

        states = self.radianToContinuous(states)
        control_inputs = self.radianToContinuous(control_inputs)
        # print(f"States shape: {states.shape} | Control inputs shape: {control_inputs.shape}")

        delta_particles = self.forward_model.forward(states, control_inputs)
        delta_particles = self.continuousToRadian(delta_particles)

        propagated_particles = self.boxplus(self.particles, delta_particles)
        self.particles = propagated_particles
        assert self.particles.shape == (control_inputs.shape[0], self.hparams["num_particles"], 3)

        return propagated_particles

    def update(self, states, measurement):
        """ Passes the current measurements to the observation model, which returns their likelihoods given the particle states. 
            The weights are then updated with these likelihoods.

        Args: 
            states (torch.tensor): 
            measurement (torch.tensor):
        """
        states = self.radianToContinuous(states)
        likelihoods = self.observation_model(states, measurement)

        if self.hparams['use_log_probs']: 
            self.weights = self.weights + likelihoods
            self.weights = self.weights - torch.logsumexp(self.weights, dim=1, keepdim=True) # normalize for valid probabilities
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



