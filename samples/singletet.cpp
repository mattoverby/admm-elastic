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
#include "AnchorForce.hpp"
#include "TetForce.hpp"
using namespace admm;
bool setup( System *system );


int main(int argc, char *argv[]){

	// Set up our solver (add nodes and forces)
	System system;
	if( !setup( &system ) ){ return 0; }

	// Initialize the solver. This creates the global matrices and
	// verifies input. It will return false on a failure.
	double timestep_s = 1.0; // unreasonably large but this is just a test
	system.settings.timestep_s = timestep_s;
	if( !system.initialize() ){ return 0; }

	// Now move the 4th node's x out and see where it ends up after one step
	system.m_x[3*3] = 200.0;
	int n_iters = 20; // Run ADMM for 20 iterations
	system.settings.admm_iters = 20;
	system.step();
	double new_x = system.m_x[3*3];

	// Print results
	std::stringstream ss;
	ss << "\n======\nSolver: ADMM, Max Iters: " << n_iters << ", Tet Force: Linear";
	ss << "\nNode 4 x: " << new_x << "\n======";
	std::cout << ss.str() << std::endl;

	return 0;
}



bool setup( System *system ){

	using namespace Eigen;
	system->settings.verbose = 0; // less prints

	// Add nodes to the system.
	// We can add the necessary values directly:
	if( false ){

		system->m_v.resize(4*3); // node velocities
		system->m_v.fill(0);
		system->m_masses.resize(4*3); // node masses (vector is scaled x3)
		system->m_masses.fill(1);

		Vector3d nodes[4];
		nodes[0] = Vector3d(0,1,0);
		nodes[1] = Vector3d(0,0,0);
		nodes[2] = Vector3d(0,0,1);
		nodes[3] = Vector3d(1,0,0);

		system->m_x.resize(4*3); // node positions
		for( int i=0; i<4; ++i ){
			for( int j=0; j<3; ++j ){ system->m_x[i*3+j] = nodes[i][j]; }
		}
	}

	// Or use the add_nodes function in System:
	else {
		VectorXd x(4*3), m(4*3); // size of node positions/masses is num nodes x3
		m.fill(1); // set node masses to 1
		x.fill(0); // use unit node positions
		x[0*3 + 1] = 1;
		x[2*3 + 2] = 1;
		x[3*3 + 0] = 1;

		// No need to add velocities, they default to zero
		system->add_nodes( x, m );
	}

	// Anchor 3 nodes
	for( int i=0; i<3; ++i ){
		std::shared_ptr<Force> af( new StaticAnchor( i ) );

		// When adding forces to the system, you can just push them on the
		// forces vector in any order.
		system->forces.push_back( af );
	}

	// Add tet force
	double stiff = 1;
	std::shared_ptr<LinearTetStrain> tf( new LinearTetStrain( 0, 1, 2, 3, stiff ) );
	system->forces.push_back( tf );

	return true;
}


