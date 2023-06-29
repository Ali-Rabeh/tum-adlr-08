"""

"""

import torch
import numpy as np

def boxplus(state, delta, scaling=1.0):
    """ Implements the boxplus operation for the SE(2) manifold. The implementation follows Lennart's manitorch code. 

    Args: 
        state (torch.tensor): A 1x1x3 tensor representing the state [x, y, theta]
        delta (torch.tensor): A 1x1x3 tensor representing the step towards the next state: [delta_x, delta_y, delta_theta]

    Returns:
        new_state (torch.tensor): A 1x1x3 tensor representing the new state: [new_x, new_y, new_theta]. Basically, new_state = state + delta,
                                  and we make sure that new_theta is between -pi and pi.
    """
    delta = scaling * delta
    new_state = state + delta
    new_state[:,:,2] = ((new_state[:,:,2] + np.pi) % (2*np.pi)) - np.pi
    return new_state

def radianToContinuous(states):
    """ Maps the state [x, y, theta_rad] to the state [x, y, cos(theta_rad), sin(theta_rad)]. 
    
    Args: 
        states (torch.tensor): A tensor with size (batch_size, num_particles, 3) representing the particles states.

    Returns: 
        continuous_states (torch.tensor): A tensor with size (batch_size, num_particles, 4) representing a continuous version of the 
                                            particle states.
    """
    continuous_states = torch.zeros(size=(states.shape[0], states.shape[1], states.shape[2]+1))
    # print(continuous_states.shape)
    continuous_states[:,:,0:2] = states[:,:,0:2]
    continuous_states[:,:,2] = torch.cos(states[:,:,2])
    continuous_states[:,:,3] = torch.sin(states[:,:,2])
    return continuous_states

def continuousToRadian(states):
    """ Maps states [x, y, cos(theta_rad), sin(theta_rad) to states [x, y, theta_rad]. 
    
    Args: 
        states (torch.tensor):

    Returns:
        discontinuous_states (torch.tensor):
    """

    discontinuous_states = torch.zeros(size=(states.shape[0], states.shape[1], states.shape[2]-1))
    discontinuous_states[:,:,0:2] = states[:,:,0:2]
    discontinuous_states[:,:,2] = torch.atan2(states[:,:,3], states[:,:,2])
    return discontinuous_states

def weightedAngularMean(angles_rad, weights):
    """ Calculates the weighted angular mean. 
        Assumptions: 
            - weights are already normalized
            - weights is a row vector
            - angles_rad is a column vector

    Args: 
        angles_rad (torch.tensor): Angles to compute the mean over, in radians.
        weights (torch.tensor): 

    Returns: 
        angular_mean (torch.tensor):
    """
    weighted_sum_cos = torch.matmul(weights, torch.cos(angles_rad))
    weighted_sum_sin = torch.matmul(weights, torch.sin(angles_rad))
    angular_mean = torch.atan2(weighted_sum_sin, weighted_sum_cos)
    return angular_mean