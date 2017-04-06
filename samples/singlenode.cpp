// Copyright (c) 2017 University of Minnesota
// 
// ADMM-Elastic Uses the BSD 2-Clause License (http://www.opensource.org/licenses/BSD-2-Clause)
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
//    conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list
//    of conditions and the following disclaimer in the documentation and/or other materials
//    provided with the distribution.
// THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF MINNESOTA, DULUTH OR CONTRIBUTORS BE 
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
// IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "System.hpp"
using namespace admm;
bool setup( System *system );


int main(int argc, char *argv[]){

	// Set up our solver (add nodes and forces)
	System system;
	if( !setup( &system ) ){ return 0; }

	// Initialize the solver. This creates the global matrices and
	// verifies input. It will return false on a failure.
	double timestep_s = 1.0; // unreasonably large but this is just a test
	system.settings.timestep_s  = timestep_s ;
	if( !system.initialize() ){ return 0; }

	// Now run the solver
	int n_iters = 20; // Let ADMM run for 20 iterations per timestep.
	system.settings.admm_iters = n_iters;
	for( int i=0; i<4; ++i ){ // Four timesteps

		system.step();

		std::cout << "step: " << i << ", pos: (" << system.m_x[0] << ", " << system.m_x[1] << ", " << system.m_x[2] << ')' << std::endl;

	}

	return 0;
}



bool setup( System *system ){

	using namespace Eigen;
	system->settings.verbose = 0;

	// Add nodes to the system.
	// We can use system->add_nodes, or just add
	// the necessary values directly:
	system->m_x.resize(3); // node positions
	system->m_x.fill(0); // set node pos to (0,0,0)
	system->m_v.resize(3); // node velocities
	system->m_v.fill(0); // set node velocities to (0,0,0)
	system->m_masses.resize(3); // node masses (vector is scaled x3)
	system->m_masses.fill(1); // mass = 1kg

	// Add one force: gravity
	std::shared_ptr< ExplicitForce > gravity( new ExplicitForce(Eigen::Vector3d(0.f,-9.8f,0.f)) );
	system->explicit_forces.push_back( gravity );

	return true;
}


