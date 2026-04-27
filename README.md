# ADMM-PD

ADMM ⊇ Projective Dynamics: Fast Simulation of Hyperelastic Models with Dynamic Constraints

[Matthew Overby](https://mattoverby.net), [George E. Brown](https://georgbrown.github.io), [Jie Li](https://www-users.cse.umn.edu/~lixx4611) and [Rahul Narain](https://www.cse.iitd.ac.in/~narain)
University of Minnesota

For the original code used in the paper, see the [admm-pd-tvcg](https://github.com/mattoverby/admm-elastic/releases/tag/v0.2-tvcg) release.

## Abstract

We apply the alternating direction method of multipliers (ADMM) optimization algorithm to implicit time integration of elastic bodies, and show that the resulting method closely relates to the recently proposed projective dynamics algorithm. However, as ADMM is a general purpose optimization algorithm applicable to a broad range of objective functions, it permits the use of nonlinear constitutive models and hard constraints while retaining the speed, parallelizability, and robustness of projective dynamics. We further extend the algorithm to improve the handling of dynamically changing constraints such as sliding and contact, while maintaining the benefits of a constant, prefactored system matrix. We demonstrate the benefits of our algorithm on several examples that include cloth, collisions, and volumetric deformable bodies with nonlinear elasticity and skin sliding effects.

## Compile and Run Examples

```sh_
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
./dillo
```

Press spacebar to start the simulation

## Citation

```
@article{overby2017admmpd, 
author={Overby, Matthew and Brown, George E. and Li, Jie and Narain, Rahul},
journal={IEEE Transactions on Visualization and Computer Graphics}, 
title={ADMM $\supseteq$ Projective Dynamics: Fast Simulation of Hyperelastic Models with Dynamic Constraints}, 
year={2017}, 
volume={23}, 
number={10}, 
pages={2222-2234}, 
doi={10.1109/TVCG.2017.2730875}, 
ISSN={1077-2626}, 
month={Oct},
}
```

## To-Do

- [ ] Cage deformation example
- [ ] More examples from paper