// Copyright (c) 2017, University of Minnesota
// 
// lbfgssolver Uses the BSD 2-Clause License (http://www.opensource.org/licenses/BSD-2-Clause)
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

#ifndef ADMM_SOLVER_H
#define ADMM_SOLVER_H 1

#include "Collider.hpp"
#include "EnergyTerm.hpp"
#include "SpringEnergyTerm.hpp"
#include "ExplicitForce.hpp"
#include "LinearSolver.hpp"

namespace admm {

// The main solver
class Solver {
public:
	typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;
	typedef Eigen::Matrix<double,3,1> Vec3;
	typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SparseMat;

	// Solver settings
	struct Settings {
		bool parse_args( int argc, char **argv ); // parse from terminal args. Returns true if help()
		void help();		// -help	print details, parse_args returns true if used
		double timestep_s;	// -dt <flt>	timestep in seconds (don't change after initialize!)
		int verbose;		// -v <int>	terminal output level (higher=more)
		int admm_iters;		// -it <int>	number of admm solver iterations
		double gravity;		// -g <flt>	force of gravity
		bool record_obj;	// -r		computes (and prints if verbose) objective value
		int linsolver;		// -ls <int>	0=LDLT, 1=NCMCGS
		double collision_w;	// -ck <flt>	collision weights (-1 = auto)
		Settings() : timestep_s(1.0/24.0), verbose(1), admm_iters(20),
			gravity(-9.8), record_obj(false), linsolver(0), collision_w(-1) {}
	};

	// RuntimeData struct used for logging.
	// Add timings are per time step.
	struct RuntimeData {
		double global_ms; // total ms for global solver
		double local_ms; // total ms for local solver
		int inner_iters; // total global step iterations
		std::vector<double> f, g; // objective values
		RuntimeData() : global_ms(0), local_ms(0), inner_iters(0) {}
		void print( const Settings &settings );
	};

	Solver();

	// Per-node (x3) data (for x, y, and z)
	VecX m_x; // node positions, scaled x3
	VecX m_v; // node velocities, scaled x3
	VecX m_masses; // node masses, scaled x3

	std::vector< std::shared_ptr<ExplicitForce> > ext_forces; // external/explicit forces
	std::vector< std::shared_ptr<EnergyTerm> > energyterms; // minimized (implicit)

	// Adds nodes to the Solver.
	// Returns the current total number of nodes after insert.
	// Assumes m is scaled x3 (i.e. 3 values per node).
	template <typename T>
	int add_nodes( T *x, T *m, int n_verts );

	// Pins vertex indices to the location indicated. If the points
	// vector is empty (or not the same size as inds), vertices are pinned in place.
	virtual void set_pins( const std::vector<int> &inds,
		const std::vector<Vec3> &points = std::vector<Vec3>() );

	// An obstacle is a passive collision object.
	virtual void add_obstacle( std::shared_ptr<PassiveCollision> obj );

	// A dynamic obstacle has vertices in m_x and is updated every frame
	virtual void add_dynamic_collider( std::shared_ptr<DynamicCollision> obj );

	// Returns true on success.
	virtual bool initialize( const Settings &settings_=Settings() );

	// Performs a Solver step
	virtual void step();

	// Returns the runtime data from the last time step.
	virtual const RuntimeData &runtime_data(){ return m_runtime; }

	// Returns the current settings
	const Settings &settings(){ return m_settings; }

protected:
	Settings m_settings; // copied from init
	RuntimeData m_runtime; // reset each iteration
	bool initialized; // has init been called?

	// Solver used in the global step
	std::shared_ptr<LinearSolver> m_linsolver;
	std::shared_ptr<Collider> m_collider;
	std::unordered_map<int,Vec3> m_pins; // vert idx -> location
	std::unordered_map<int, std::shared_ptr<SpringPin> > m_pin_energies;

	// Global matrices
	SparseMat m_D, m_Dt; // reduction matrix
	VecX m_W_diag; // diagonal of the weight matrix

	// Solver variables computed in initialize
	SparseMat solver_termA;
	SparseMat solver_Dt_Wt_W;

}; // end class Solver


template <typename T>
int Solver::add_nodes( T *x, T *m, int n_verts ){
	int prev_n = m_x.size();
	int n_verts3 = n_verts*3;
	m_x.conservativeResize(prev_n+n_verts3);
	m_v.conservativeResize(prev_n+n_verts3);
	m_masses.conservativeResize(prev_n+n_verts3);
	for( int i=0; i<n_verts3; ++i ){
		int idx = prev_n+i;
		m_x[idx] = x[i];
		m_v[idx] = 0.0;
		m_masses[idx] = m[i];
	}
	return (prev_n+n_verts3)/3;
}


} // end namespace admm

#endif




