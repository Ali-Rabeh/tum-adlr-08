import os
import torch

# dataset used for training
train_path_list = [
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=0.000_s=0.000_t=0.000.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=0.000_s=0.200_t=0.698.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=0.000_s=0.400_t=1.047.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=0.000_s=0.600_t=-0.349.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=0.000_s=0.800_t=-1.047.json"), 
    
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=1.000_s=0.100_t=0.698.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=1.000_s=0.300_t=1.396.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=1.000_s=0.500_t=0.000.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=1.000_s=0.700_t=-0.698.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=1.000_s=0.900_t=-1.396.json"), 
    
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=2.000_s=0.000_t=0.349.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=2.000_s=0.200_t=1.047.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=2.000_s=0.400_t=0.698.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=2.000_s=0.600_t=-0.349.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=2.000_s=0.800_t=-0.698.json"), 
    
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=3.000_s=0.100_t=1.047.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=3.000_s=0.300_t=0.698.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=3.000_s=0.500_t=1.396.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=3.000_s=0.700_t=-0.349.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=3.000_s=0.900_t=0.000.json")
]

train_mean = torch.tensor([[-9.4558e-05,  2.6232e-05, -4.7562e-04,  2.6877e-06, -6.0178e-07, 9.0652e-06, -1.2292e-02, -6.6984e-03, -4.0082e-05, -9.4513e-05, 2.6231e-05, -4.7591e-04]])
train_std = torch.tensor([[8.4274e-05, 9.4910e-05, 1.9013e-03, 6.9006e-06, 6.9973e-06, 1.5161e-04, 1.1808e-02, 1.6012e-02, 2.6409e-05, 8.4421e-05, 9.4990e-05, 1.9037e-03]])

train_min = torch.tensor([[-3.1813e-02, -3.7792e-02, -4.6015e-01, -2.4350e-03, -2.4369e-03,
         -3.6652e-04, -2.5382e+00, -8.2470e+00, -1.0860e-02, -3.1813e-02,
         -3.7792e-02, -4.6015e-01]])
train_max = torch.tensor([[1.2503e-02, 5.0536e-02, 4.9215e-01, 2.3730e-03, 2.4119e-03, 2.9671e-04,
         3.9173e+00, 6.0686e+00, 1.1036e-02, 1.2503e-02, 5.0536e-02, 4.9215e-01]])

# dataset used for validation
validation_path_list = [
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=0.000_s=0.300_t=0.000.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=0.000_s=0.500_t=0.698.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=0.000_s=0.700_t=1.396.json"),  

    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=1.000_s=0.000_t=0.698.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=1.000_s=0.400_t=1.047.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=1.000_s=0.600_t=1.396.json"),

    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=2.000_s=0.500_t=0.698.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=2.000_s=0.700_t=-0.698.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=2.000_s=0.900_t=0.000.json"),   

    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=3.000_s=0.200_t=0.000.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=3.000_s=0.400_t=0.349.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=3.000_s=0.600_t=1.047.json")
]

validation_mean = torch.tensor([[ 6.9963e-05,  3.8133e-05,  4.1854e-03, -2.5945e-06, -9.8929e-07, -2.2903e-04, -3.9136e-03, -1.5550e-03, -5.8680e-06,  6.9881e-05, 3.8095e-05,  4.1861e-03]])
validation_std = torch.tensor([[1.3264e-04, 7.8608e-05, 7.3596e-03, 8.5618e-06, 5.4759e-06, 1.8754e-03, 1.6010e-02, 1.3043e-02, 3.0046e-05, 1.3282e-04, 7.8705e-05, 7.3589e-03]])

validation_min = torch.tensor([[-3.2065e-02, -2.8855e-02, -5.1584e-01, -2.3710e-03, -2.3313e-03,
         -1.5708e-04, -3.6884e+00, -5.3868e+00, -2.3036e-02, -3.2065e-02,
         -2.8855e-02, -5.1584e-01]])
validation_max = torch.tensor([[3.7603e-02, 3.6424e-02, 5.9090e+00, 2.0580e-03, 2.4231e-03, 2.7925e-04,
         4.3158e+00, 7.7526e+00, 7.7433e-03, 3.7603e-02, 3.6424e-02, 5.9090e+00]])

# dataset used for testing
test_path_list = [
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=0.000_s=0.100_t=0.000.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=0.000_s=0.900_t=0.698.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=0.000_s=1.000_t=1.047.json"),  

    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=1.000_s=0.200_t=-0.349.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=1.000_s=0.800_t=-1.396.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=1.000_s=1.000_t=0.698.json"),

    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=2.000_s=0.100_t=0.698.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=2.000_s=0.300_t=-1.047.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=2.000_s=1.000_t=0.349.json"),   

    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=3.000_s=0.000_t=0.000.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=3.000_s=0.200_t=-0.349.json"), 
    os.path.abspath("../pushdata/plywood/rect1/motion_surface=plywood_shape=rect1_a=0_v=100_i=3.000_s=1.000_t=1.047.json") 
]

test_mean = torch.tensor([[ 1.4125e-04, -9.9369e-05, -5.8108e-04, -3.9843e-06,  3.3249e-06, 1.2629e-05, -6.2884e-03,  6.1137e-03, -5.5610e-06,  1.4126e-04, -9.9368e-05, -5.8114e-04]])
test_std = torch.tensor([[1.7295e-04, 1.1925e-04, 2.2736e-03, 1.1160e-05, 7.8305e-06, 1.6352e-04, 1.7238e-02, 1.5586e-02, 2.7105e-05, 1.7295e-04, 1.1926e-04, 2.2738e-03]])

test_min = torch.tensor([[-3.0546e-02, -3.5163e-02, -6.0163e-01, -2.3260e-03, -2.2722e-03,
         -1.7453e-04, -4.7100e+00, -3.1850e+00, -7.2725e-03, -3.0546e-02,
         -3.5163e-02, -6.0163e-01]])
test_max = torch.tensor([[4.0568e-02, 1.9610e-02, 4.8599e-01, 1.7130e-03, 2.3992e-03, 2.0944e-04,
         3.3518e+00, 6.1078e+00, 7.3675e-03, 4.0568e-02, 1.9610e-02, 4.8599e-01]])