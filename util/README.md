# Utilities

This directory contains helper files for other parts of the project. Specifically, the individual files contain: 

- `data_file_paths.py`: Contains hardcoded paths towards the individual .json files making up our dataset. These paths only work if you use the recommended setup when working with the MIT dataset which is described [here](https://web.mit.edu/mcube//push-dataset/). Also contains the necessary constants to normalize or standardize the train, validation and test datasets. 

- `dataset.py`: Contains class definitions implementing the dataset structure we used when training and evaluating the differentiable particle filters. 

- `manifold_helpers.py`: Contains functions for handling state estimation in the filters on the SE2-manifold and for transforming orientation estimates into continuous representations and from these back to "normal" radian representations.

- `normalize_datasets.py`: Small helper script used to calculate the normalization and standardization constants for our dataset. 

- `test_dataset.py`: Small helper script to check that all trajectories in the dataset are ok based on a visual evaluation and that recorded forces and trajectory correspond to each other. This was only used in the very beginning of the project.   