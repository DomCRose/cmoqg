# Classical metastability in an open quantum glass

This repository contains code that can be used to generated the results in the paper [Hierarchical classical metastability in an open quantum East model](https://arxiv.org/abs/2010.15304), along with example notebooks for a small system.

The notebooks demonstrate:
* dqe_jump_trajectories: generating quantum jump Monte-Carlo trajectories for the open quantum east model.
* dqe_metastability: using the generic algorithm described in the paper to construct the classical metastable manifold for a 3 spin open quantum east system.
* dqe_metastability_with_symmetries: using the symmetry-aware algorithm described in the paper to construct the classical metastable manifold.

This is achieved by utilizing the code in the source and models folders:
* cyclic_representations: contains code for constructing projectors onto the translation symmetry eigenspaces for spin-1/2 systems, along with some other utilities.
* master_operators: contains generic code for constructing matrix representations of master operators, with the potential to add biases to study large deviations or to utilize weak symmetries. The weakly symmetric master operator assumes there is a single cyclic symmetry.
* quantum_jumps: a custom implementation of quantum jump Monte-Carlo (likely slower than QuTiP).
* random_linear_algebra: contains some utility functions for constructing random states and random orthonormal matrices.
* metastability: contains all code related to the construction of classical metastable manifolds using eigenspectrum data, along with functions to process the resulting simplex into extreme metastable state and corresponding probability operators.
* dissipative_quantum_east: contains classes which inherit from the master_operators classes to construct matrix representations of the Lindblad operator for the open quantum East model, with classes that allow to add biases and utilize symmetries.
