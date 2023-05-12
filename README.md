# TUM-ADLR-08 - Multimodal Sensor Fusion in Differentiable Filters Using A Learned Proposal Distribution
Repository for project of Advanced Deep Learning for Robotics (ID 08)

## Milestones 
The key steps for implementing the project:
- Implement a differentiable filter for estimating the state of a pushed object.
- Adapt the Differentiable Particle Proposal Filter from [here]() to a pushing scenario.
- Fuse tactile, proprioceptive and visual measurements in the learned proposal distribution in a novel way such as in a single network.

## Repository Overview
| Directory/Script | Content |
| ---------------- | ------- |
| docs             | Contains all handed-in files during the project. |
| experiments      | Contains scripts for training and evaluating differentiable filters with varying data? | 
| models           | Contains our implementations of differentiable filters. |

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