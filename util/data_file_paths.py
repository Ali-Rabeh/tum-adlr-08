import os

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