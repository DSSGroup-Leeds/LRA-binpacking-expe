This repository contains all material used to perform the experiments presented in the paper "Affinity-Aware Resource Provisioning for Long-Running Applications in Shared Clusters".


Input instances generation
==========================

The Python scripts to format and generate instances from the TClab dataset are in `data/scripts`.
The scripts needs the 3 following CSV files from the TClab dataset (available in `data/TClab`:
- `app_resources.csv`
- `app_interference.csv`
- `instance_deployment.csv`

The datasets `TClab_dataset_2D.csv` and `TClab_dataset_TS.csv` are generated from these 3 files using the script `generate_TClab_dataset.py`.
These datasets serve as baselines to generate all instances of higher density and large scale, both for fixed and time-varying resource requirements, using the scripts `generate_higher_density.py` and `generate_large_scale.py`.


Building executables
====================

The only dependency for building executables is cmake.
We use [Nix](https://nixos.org/) as a package manager.
`default.nix` file contains a recipe to build all executables needed.
With Nix installed, simply run `nix-build default.nix -A binpack` and nix will do the rest and create a `result/bin` folder with all built executables.

If you want to be in an environment to hack the C++ algorithms and manually build the executables, run `nix-shell default.nix -A binpack` instead.


Input file format
=================

The input datasets are TAB-separated CSV files containing 6 columns.
Apart from the header line, each line in a file describes an application.
The 6 columns are:
- `app_id`: The unique identifier of the application
- `nb_instances`: The number of replicas of the application
- `core`: The core requirement of the application
- `memory`: The memory requirement of the application
- `inter_degree`: The number of affinity restrictions of the application
- `inter_aff`: The list of affinity restrictions, formated as a string representation of a Python list. The list must have `inter_degree` elements, and each element is a couple `(application_id, affinity_value)` (using notations of the paper, this corresponds to the couples $(j, a_{ij})$ for application $i$)



Output file format
==================

Each executable generates an output file in `data/results`, which is a TAB-separated CSV file.
Apart from the header line, each line in a file records the results obtained by each algorithm applied to a given dataset instance.
The columns are:
- `instance_name`: The name of the dataset instance
- `LB`: The value of the lower bound
- `best_sol` and `best_algo`: The best solution value found, and the algorithm name that found it. If multiple algorithms found the best solution, only the first algorithm is kept.
- One column per algorithm name, with the solution value found by this algorithm
- One column per algorithm name and suffixed with `_time`, with the time (in seconds) taken by the algorithm to find the solution

