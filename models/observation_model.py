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

class ObservationModelImages(nn.Module):
    def __init__(self, use_log_probs):
        """This will output weights for the particles.
        
        """
        super().__init__()

        if use_log_probs:
            last_layer = nn.Identity()
        else:
            last_layer = nn.Sigmoid()

        # define structure for image encoding
        self.image_encoder = nn.Sequential(
            # (128, 128, 1)
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=8), 
            nn.ReLU(),

            # (63, 63, 8)
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),

            # (31, 31, 8)
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),

            # (15, 15, 8)
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),

            # (7, 7, 8)
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),

            # (3. 3. 1)
        )

        self.small_mlp = nn.Sequential(
            nn.Linear(9+4, 3),
            nn.ReLU(),
            nn.Linear(3, 1), 
            last_layer
        )

    def forward(self, particle_states, image):
        """Calculates the likelihoods of current measurements given the particle states. 

        Args: 
            particle_states (torch.tensor): A tensor with size (batch_size, num_particles, 4) representing the particle states 
                                            in the continuous representation.
            image (torch.tensor):
        """

        num_particles = particle_states.shape[1]
        state_dim = particle_states.shape[2]

        # pass image through the encoder and reshape it for the subsequent linear layers
        encoded_image = self.image_encoder(image)
        encoded_image = torch.reshape(encoded_image, (1,-1))
        encoded_image_length = encoded_image.shape[1]

        # to compare it to the particle states, reshape the encoded image to 
        encoded_image_repeated = torch.zeros(size=(1, num_particles, encoded_image_length))
        encoded_image_repeated[0,:,:] = encoded_image[0,:].repeat(num_particles,1)
        assert encoded_image_repeated.shape == (1, num_particles, encoded_image_length) 

        network_inputs = torch.concat((particle_states, encoded_image_repeated), dim=2)
        assert network_inputs.shape == (1, num_particles, state_dim+encoded_image_length)

        likelihoods = self.small_mlp(network_inputs)
        assert likelihoods.shape == (1, num_particles, 1)

        return likelihoods

# class ObservationModelForcesAndImages(nn.Module):
#     def __init__(self, use_log_probs):
#         """This will output weights for the particles.
        
#         """
#         super().__init__()

#         if use_log_probs:
#             last_layer = nn.Identity()
#         else:
#             last_layer = nn.Sigmoid()

#         # define structure for image encoding
#         self.image_encoder = nn.Sequential(
#             # (128, 128, 1)
#             nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2),
#             nn.BatchNorm2d(num_features=8), 
#             nn.ReLU(),

#             # (63, 63, 8)
#             nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2),
#             nn.BatchNorm2d(num_features=8),
#             nn.Relu(),

#             # (31, 31, 8)
#             nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2),
#             nn.BatchNorm2d(num_features=8),
#             nn.ReLU(),

#             # (15, 15, 8)
#             nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2),
#             nn.BatchNorm2d(num_features=8),
#             nn.ReLU(),

#             # (7, 7, 8)
#             nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=2),
#             nn.BatchNorm2d(num_features=1),
#             nn.ReLU(),

#             # (3. 3. 1)
#         )

#         # define structure for force encoding
#         self.force_encoder = nn.Sequential(

#         )

#     def forward(self, particle_states, measurements):
#         """Calculates the likelihoods of current measurements given the particle states. 

#         Args: 
#             particle_states (torch.tensor): A tensor with size (batch_size, num_particles, 4) representing the particle states 
#                                             in the continuous representation.
#             measurements (torch.tensor): The current measured forces/torques.
#         """
#         num_input_points = measurements.shape[0]
#         num_particles = particle_states.shape[1]

#         state_dim = particle_states.shape[2]
#         measurements_dim = measurements.shape[1]

#         measurements_reshaped = torch.zeros(size=(num_input_points, num_particles, measurements_dim))
#         for n in range(num_input_points):
#             measurements_reshaped[n,:,:] = measurements[n,:].repeat(num_particles,1)
#         assert measurements_reshaped.shape == (num_input_points, num_particles, measurements_dim)

#         network_inputs = torch.concat((particle_states, measurements_reshaped), dim=2)
#         assert network_inputs.shape == (num_input_points, num_particles, state_dim+measurements_dim)

#         likelihoods = self.model(network_inputs)
#         assert likelihoods.shape == (num_input_points, num_particles, 1)

#         return likelihoods