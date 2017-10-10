# NOTE

This code is a bit outdated and I'll be modifying it to include material from our recent [IEEE TVCG paper](http://ieeexplore.ieee.org/document/7990052). Also, I believe there to be some bugs. If you plan to use this code for comparison, I ask that you please wait little while for me to fix up some issues.

# admm-elastic

[ADMM ⊇ Projective Dynamics: Fast Simulation of General Constitutive Models](http://rahul.narain.name/admm-pd/)  
[Rahul Narain](http://rahul.narain.name/), [Matthew Overby](http://www.mattoverby.net/), and [George E. Brown](http://www-users.cs.umn.edu/~brow2327/)  
University of Minnesota

Please see [admm-elastic-samples](http://www.github.com/mattoverby/admm-elastic-samples/) for examples used in the paper.

# abstract

We apply the alternating direction method of multipliers (ADMM) optimization algorithm to implicit time integration of elastic bodies, and show that the
resulting method closely relates to the recently proposed projective dynamics algorithm. However, as ADMM is a general-purpose optimization algorithm applicable
to a broad range of objective functions, it permits the use of nonlinear constitutive models and hard constraints while retaining the speed, parallelizability, and
robustness of projective dynamics. We demonstrate these benefits on several examples that include cloth, collisions, and volumetric deformable bodies with nonlinear elasticity.

# citation

BibTex:  

	@inproceedings{Narain2016,
	 author = {Narain, Rahul and Overby, Matthew and Brown, George E.},
	 title = {{ADMM} $\supseteq$ Projective Dynamics: Fast Simulation of General Constitutive Models},
	 booktitle = {Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation},
	 series = {SCA '16},
	 year = {2016},
	 isbn = {978-3-905674-61-3},
	 location = {Zurich, Switzerland},
	 pages = {21--28},
	 numpages = {8},
	 url = {http://dl.acm.org/citation.cfm?id=2982818.2982822},
	 acmid = {2982822},
	 publisher = {Eurographics Association},
	 address = {Aire-la-Ville, Switzerland, Switzerland},
	} 

# installation

The best way to include admm-elastic in your project is to include it in another directory,
use cmake to call add_subdirectory, then use the ${ADMME_LIBRARIES} and
${ADMME_INCLUDE_DIRS} set as parent variables during the build.

# license

Copyright (c) 2017 University of Minnesota

ADMM-Elastic Uses the BSD 2-Clause License (http://www.opensource.org/licenses/BSD-2-Clause)  
Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:  
1. Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.  
2. Redistributions in binary form must reproduce the above copyright notice, this list
of conditions and the following disclaimer in the documentation and/or other materials
provided with the distribution.  
THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF MINNESOTA, DULUTH OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
