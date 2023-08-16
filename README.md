# TUM-ADLR-08 - Multimodal Sensor Fusion in Differentiable Filters Using A Learned Proposal Distribution
Welcome to the repository for our project in the SS23 iteration of the Advanced Deep Learning for Robotics (ADLR) course at TUM! We were designated as group 8 and explored methods for sensor fusion in differentiable particle filters to enhance nonlinear state estimates in a real-world pushing scenario. 

## Repository Overview
| Directory/Script | Content |
| ---------------- | ------- |
| data | Contains the dataset we used during our project as jsons. The data is a subset of the MIT Push dataset and you can find more information on this dataset [here](https://web.mit.edu/mcube//push-dataset/). |
| doc | Contains all handed-in files during the project as well as some additional insightful figures. |
| models | Contains our implementations of a differentiable particle filter and different forward and observation models to plug into the particle filter. Also contains saved versions of the already trained models we used for producing our final results. |
| output | Contains figures and animations produced when running `run_filter.py`. | 
| util | Contains helper functions and scripts used during the course of the project. Look inside for a more detailed overview. |
| `camera_helpers.py` | Helper script to generate synthetic camera images from a current object state. |
| `run_filter.py` | Script for evaluating an already trained filter. For more details on how to use it, see below. |
| `train.py`| Script for training a differentiable particle filter. For more details on how to use it, see below. |

## How to use the project 
### I. Training
The script `train.py` is used for training a differentiable particle filter. Training behavior can be controlled via the hyperparameter dictionary right at the start of the script. Once you have set the parameters to your liking, run

`python3 train.py`

from the main directory of the repository.

### II. Inference/Testing
An already trained filter can be evaluated on the testset with the `run_filter.py` script. The behavior can be again controlled by the hyperparameter dictionary at the start of the script. Once these are set, run 

`python3 run_filter.py`

from the main directory of the repository. This will put both figures and animations (if desired) into the [output/figures]() and [output/animations]() to visualize the chosen filter's behavior. 

## Milestones 
The key steps for implementing the project were:
- :white_check_mark: Implement a differentiable filter for estimating the state of a pushed object.
- :white_check_mark: Adapt the Differentiable Particle Proposal Filter from [here]() to a pushing scenario.
- :white_check_mark: Fuse tactile, proprioceptive and visual measurements via both a learned proposal distribution and a combined observation model and evaluate their respective performances.

## Helpful Links
### Paper
- [How to train your Differentiable Filter](https://arxiv.org/abs/2012.14313): Presents an overview on how to train differentiable filters and compares them on two tasks.
- [Multimodal Sensor Fusion with Differentiable Filters](https://arxiv.org/abs/2010.13021): Explores different architectures for fusing multimodal information with differentiable filters.  
- [Learning a State-Estimator for Tactile In-Hand Manipulation](https://ieeexplore.ieee.org/document/9981730): Implements a differentiable particle filter for estimating the pose of a manipulated object using only tactile information. 
- [More than A Million Ways to Be Pushed](https://arxiv.org/abs/1604.04038): Presents the MIT Push dataset used in this project.

### Other Open Source Code for Differentiable Filters
- [torchfilter](https://github.com/stanford-iprl-lab/torchfilter) library: Differentiable implementations of common Bayesian filters written in pytorch.  
- [multimodalfilter](https://github.com/brentyi/multimodalfilter): Pytorch code used in the multimodal fusion paper. 

### Data
The MIT Push dataset can be downloaded from the MIT MCube Lab website [here](https://web.mit.edu/mcube//push-dataset/). Scripts for preprocessing and rendering that data can be found [here](https://github.com/mcubelab/pdproc). 
Dataset used for training, testing and validation can be found under the data folder of this repository.
