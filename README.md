# admm-elastic

[ADMM ⊇ Projective Dynamics: Fast Simulation of Hyperelastic Models with Dynamic Constraints](http://www-users.cs.umn.edu/~over0219/pages/admmpd_abstract.html)  
[Matthew Overby](http://www.mattoverby.net/), [George E. Brown](http://www-users.cs.umn.edu/~brow2327/),
[Jie Li](http://www-users.cs.umn.edu/~lixx4611/) and [Rahul Narain](http://rahul.narain.name/)  
University of Minnesota

Materials from the TVCG paper will be added incrementally (samples/solvers/etc...)

# notes

This project contains submodules:
- [mclscene](https://github.com/mattoverby/mclscene)
- [mcloptlib](https://github.com/mattoverby/mcloptlib)

and has the dependences:
- Eigen3
- GLFW
- OpenGL

Todo:
- [Stable Neo-Hookean](http://graphics.pixar.com/library/StableElasticity)
- UzawaCG for self-collisions
- Penalty collisions for self-collisions
- Slide constraints
- Bending force
- Demos from SCA/TVCG papers
- More/better tests

# abstract

We apply the alternating direction method of multipliers (ADMM) optimization algorithm to implicit time integration of elastic bodies,
and show that the resulting method closely relates to the recently proposed projective dynamics algorithm. However, as ADMM is a general
purpose optimization algorithm applicable to a broad range of objective functions, it permits the use of nonlinear constitutive models and
hard constraints while retaining the speed, parallelizability, and robustness of projective dynamics. We further extend the algorithm to
improve the handling of dynamically changing constraints such as sliding and contact, while maintaining the benefits of a constant,
prefactored system matrix. We demonstrate the benefits of our algorithm on several examples that include cloth, collisions, and volumetric
deformable bodies with nonlinear elasticity and skin sliding effects.

# citation

BibTex:  

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

