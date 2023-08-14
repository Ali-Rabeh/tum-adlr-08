import numpy as np
import cv2
import torch

from models.observation_model import ObservationModelImages
from util.manifold_helpers import radianToContinuous

class ImageGenerator:
    def __init__(self, object_size=[0.045, 0.045], image_size=(128,128), dist_to_surface_m=0.4, fov_x_deg=60, fov_y_deg=60): 
        self.object_size = object_size
        self.image_size = image_size

        self.dist_to_surface_m = dist_to_surface_m
        self.fov_x_deg = fov_x_deg
        self.fov_y_deg = fov_y_deg  

    def get_corner_positions(self, se2_pose_center):
        """ Calculates the corner positions of the object we push given the pose of its center. 
            This assumes the object has a square top surface with known dimensions.

        Args: 
            se2_pose_center (torch.tensor): A (1 x 3) tensor representing the pose of the object's center. 

        Returns: 
            corner_positions (torch.tensor): A (4 x 2) tensor holding the positions of the object's corners.

        """
        object_diameter = np.sqrt(self.object_size[0]**2+self.object_size[1]**2)

        # the corner names in the following refer to the unrotated object
        pos_upper_right = se2_pose_center[0:2] + object_diameter/2.0 * torch.tensor([np.cos(0.25*np.pi + se2_pose_center[2]), np.sin(0.25*np.pi + se2_pose_center[2])])
        pos_upper_right = pos_upper_right[None,:]

        pos_upper_left = se2_pose_center[0:2] + object_diameter/2.0 * torch.tensor([np.cos(0.75*np.pi + se2_pose_center[2]), np.sin(0.75*np.pi + se2_pose_center[2])])
        pos_upper_left = pos_upper_left[None,:]

        pos_lower_left = se2_pose_center[0:2] + object_diameter/2.0 * torch.tensor([np.cos(1.25*np.pi + se2_pose_center[2]), np.sin(1.25*np.pi + se2_pose_center[2])])
        pos_lower_left = pos_lower_left[None,:]

        pos_lower_right = se2_pose_center[0:2] + object_diameter/2.0 * torch.tensor([np.cos(1.75*np.pi + se2_pose_center[2]), np.sin(1.75*np.pi + se2_pose_center[2])])
        pos_lower_right = pos_lower_right[None,:]

        return torch.cat((pos_upper_right, pos_upper_left, pos_lower_left, pos_lower_right), dim=0)

    def cartesian_to_pixel_coordinates(self, points): 
        """ Transforms a collection of 2d points in cartesian space into pixel parameters given some camera parameters and camera position.  

        Args: 
            points: A (num_points x 2) tensor containing the points you want to transform into pixel coordinates.

        Returns: 
            pixel_coordinates (torch.tensor): A (num_points x 2) tensor containing the pixel coordinates corresponding to the points. 

        """
        # print(f"Corner positions: {points}")

        fov_x_m = 2.0 * self.dist_to_surface_m * torch.tan(torch.tensor(self.fov_x_deg/2.0)*np.pi/180.0)
        fov_y_m = 2.0 * self.dist_to_surface_m * torch.tan(torch.tensor(self.fov_y_deg/2.0)*np.pi/180.0)

        # TODO: check if there's a +1 missing somewhere
        # print(f"Argument: {((points[:,0]+fov_x_m/2.0)/fov_x_m) * self.image_size[1]}")
        pixel_coords_col = torch.round(((points[:,0]+fov_x_m/2.0)/fov_x_m) * self.image_size[1]).to(torch.int32)
        pixel_coords_row = torch.round(self.image_size[0] - ((points[:,1]+fov_y_m/2.0)/fov_y_m)*self.image_size[0]).to(torch.int32)

        # add another dimension
        pixel_coords_col = pixel_coords_col[..., np.newaxis]
        # print(pixel_coords_col)
        pixel_coords_row = pixel_coords_row[..., np.newaxis]

        assert ((pixel_coords_col >= 1) & (pixel_coords_col <= self.image_size[1])).all()
        assert ((pixel_coords_row >= 1) & (pixel_coords_row <= self.image_size[0])).all()

        return np.concatenate((pixel_coords_col, pixel_coords_row), axis=1)

    def generate_image(self, se2_pose_center):
        """ Generates a 128 x 128 binary image of the object's current state. 
            This assumes we know the camera parameters, camera position and object dimensions. 

        Args: 
            se2_pose_center (torch.tensor): A (1 x 3) tensor representing the pose of the object's center. 

        Returns: 
            image (np.array): A binary image of the current state of the object in a format compatible with OpenCV. 

        """
        corner_points = self.get_corner_positions(se2_pose_center) 
        # print(f"Corner Points: {corner_points}")
        corner_pixels = self.cartesian_to_pixel_coordinates(corner_points)
        # print(f"Corner Pixels: {corner_pixels}")

        image = np.zeros(shape=(self.image_size), dtype=np.uint8)
        image = cv2.fillPoly(image, [corner_pixels], color=255)
        return image

def main():
    se2_poses_center = torch.tensor([[0.0, 0.0, -np.pi/4.0],
                                     [-0.05, 0.05, 0.0], 
                                     [0.05, 0.05, 0.0], 
                                     [0.05, -0.05, 0.0], 
                                     [-0.05, -0.05, 0.0]])

    # se2_poses_center = torch.tensor([[ 1.0822e-06, -1.4529e-07,  2.0146e-05],
    #     [ 5.6922e-06, -9.2387e-07,  2.2888e-05],
    #     [ 9.7472e-06, -4.0233e-06,  4.7684e-06],
    #     [ 8.8103e-06,  2.6077e-08, -5.9605e-08],
    #     [ 6.7558e-04,  2.9241e-04, -1.2771e-02],
    #     [ 2.5236e-03,  9.3730e-04, -4.5247e-02],
    #     [ 3.7917e-03,  1.2288e-03, -6.7038e-02],
    #     [ 4.7905e-03,  1.4019e-03, -8.5374e-02],
    #     [ 6.2534e-03,  1.7132e-03, -1.1167e-01],
    #     [ 7.5307e-03,  1.8996e-03, -1.3720e-01],
    #     [ 8.6936e-03,  2.0829e-03, -1.5724e-01],
    #     [ 1.0244e-02,  2.3748e-03, -1.8635e-01],
    #     [ 1.1561e-02,  2.6153e-03, -2.1097e-01],
    #     [ 1.2729e-02,  2.8222e-03, -2.3417e-01],
    #     [ 1.3982e-02,  3.0478e-03, -2.5911e-01],
    #     [ 1.5245e-02,  3.2385e-03, -2.8343e-01],
    #     [ 1.6405e-02,  3.3985e-03, -3.0632e-01],
    #     [ 1.7617e-02,  3.6099e-03, -3.3926e-01],
    #     [ 1.8102e-02,  3.6799e-03, -3.4990e-01],
    #     [ 2.0007e-02,  3.9453e-03, -3.8953e-01],
    #     [ 2.1101e-02,  4.0122e-03, -4.1455e-01],
    #     [ 2.2241e-02,  4.1651e-03, -4.3987e-01],
    #     [ 2.3501e-02,  4.3346e-03, -4.7050e-01],
    #     [ 2.4752e-02,  4.5868e-03, -4.9900e-01],
    #     [ 2.5833e-02,  4.8300e-03, -5.2340e-01],
    #     [ 2.6905e-02,  5.0917e-03, -5.4756e-01],
    #     [ 2.7960e-02,  5.3371e-03, -5.7081e-01],
    #     [ 2.8749e-02,  5.5212e-03, -5.8822e-01],
    #     [ 2.9229e-02,  5.6653e-03, -5.9960e-01],
    #     [ 2.9316e-02,  5.6977e-03, -6.0156e-01],
    #     [ 2.9307e-02,  5.6941e-03, -6.0160e-01],
    #     [ 2.9315e-02,  5.6946e-03, -6.0149e-01],
    #     [ 2.9306e-02,  5.6943e-03, -6.0150e-01],
    #     [ 2.9303e-02,  5.6957e-03, -6.0154e-01],
    #     [ 2.9318e-02,  5.6961e-03, -6.0155e-01],
    #     [ 2.9311e-02,  5.6921e-03, -6.0148e-01],
    #     [ 2.9318e-02,  5.6959e-03, -6.0157e-01],
    #     [ 2.9303e-02,  5.6960e-03, -6.0152e-01],
    #     [ 2.9312e-02,  5.6979e-03, -6.0153e-01],
    #     [ 2.9312e-02,  5.6954e-03, -6.0163e-01],
    #     [ 2.9309e-02,  5.6929e-03, -6.0152e-01],
    #     [ 2.9317e-02,  5.6907e-03, -6.0150e-01],
    #     [ 2.9311e-02,  5.6961e-03, -6.0148e-01],
    #     [ 2.9308e-02,  5.6954e-03, -6.0150e-01],
    #     [ 2.9314e-02,  5.6965e-03, -6.0146e-01],
    #     [ 2.9305e-02,  5.6952e-03, -6.0153e-01],
    #     [ 2.9305e-02,  5.6952e-03, -6.0153e-01],
    #     [ 2.9305e-02,  5.6952e-03, -6.0153e-01],
    #     [ 2.9305e-02,  5.6952e-03, -6.0153e-01]])

    # video_path = "test.avi"
    # video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps=10, frameSize=image_size)

    observation_model = ObservationModelImages(use_log_probs=True)
    image_generator = ImageGenerator()

    for n in range(se2_poses_center.shape[0]): 
        print(f"SE2 pose center shape: {se2_poses_center[n,:].shape}")
        image = image_generator.generate_image(se2_poses_center[n,:])

        torch_image = torch.tensor(image, dtype=torch.float32)
        print(f"Image type: {torch_image.dtype}")

        torch_image = torch_image[None, None, :, :]
        print(f"Image shape: {torch_image.shape}") # (1, 1, 128, 128) --> conv layers expect (batch_size, channels, height, width)

        current_pose = se2_poses_center[n,:]
        current_pose = torch.tensor(current_pose[None, None, ...])
        print(f"SE2 pose center shape: {current_pose.shape}")
        likelihoods = observation_model.forward(radianToContinuous(current_pose), torch_image)
        print(f"Likelihoods: {likelihoods}")

        # image = cv2.merge([image, image, image])
        # video_writer.write(image)

        cv2.imshow("Test image", image)
        cv2.waitKey(0)

    # video_writer.release()    
    cv2.destroyAllWindows()

if __name__ == '__main__': 
    main()
