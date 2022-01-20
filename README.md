This repository contains all material used to perform the experiments presented in the paper.


Input instances generation
==========================

The python scripts to format and generate instances from the TClab dataset are in `data/scripts`.
The scripts needs the 3 following CSV files from the TClab dataset:
    - `app_resources.csv`
    - `app_interference.csv`
    - `instance_deployment.csv`


First, the script `generate_TClab_dataset.py` should be run to format the base datasets from the TClab files.
Then, the scripts `generate_higher_density.py` and `generate_large_scale.py` can be run to generate all instances of higher density and large scale, both for fixed and time-varying resource requirement.


Building executables
====================

We use [Nix](https://nixos.org/) as a package manager.
`default.nix` file contains a recipe to build all executables needed.
With Nix installed, simply run `nix-build default.nix -A binpack` and nix will do the rest and create a `result/bin` folder with all built executables.

If you want to be in an environment to hack the C++ algorithms and manually build the executables, run `nix-shell default.nix -A binpack` instead.
