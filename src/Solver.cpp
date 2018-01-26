// Copyright (c) 2017, University of Minnesota
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

#include "Solver.hpp"
#include "MCL/MicroTimer.hpp"
#include <fstream>
#include <unordered_set>
#include "NodalMultiColorGS.hpp"
#include <unordered_map>

using namespace admm;
using namespace Eigen;


void Solver::step(){
	if( m_settings.verbose > 0 ){
		std::cout << "\nSimulating with dt: " <<
		m_settings.timestep_s << "s..." << std::flush;
	}

	mcl::MicroTimer t;

	// Other const-variable short names and runtime data
	const int dof = m_x.rows();
	const int n_nodes = dof/3;
	const double dt = m_settings.timestep_s;
	const int n_energyterms = energyterms.size();
	const int n_threads = std::min(n_energyterms, omp_get_max_threads());
	m_runtime = RuntimeData(); // reset

	// Take an explicit step to get predicted node positions
	// with simple forces (e.g. wind).
	const int n_ext_forces = ext_forces.size();
	for( int i=0; i<n_ext_forces; ++i ){ ext_forces[i]->project( dt, m_x, m_v, m_masses ); }

	// Add gravity
	if( std::abs(m_settings.gravity)>0 ){
		for( int i=0; i<n_nodes; ++i ){ m_v[i*3+1] += dt*m_settings.gravity; }
	}

	// Position without elasticity/constraints
	VecX x_bar = m_x + dt * m_v;
	VecX M_xbar = m_masses.asDiagonal() * x_bar;
	VecX curr_x = x_bar; // Temperorary x used in optimization

	// Initialize ADMM vars
	VecX curr_z = m_D*m_x;
	VecX curr_u = VecX::Zero( curr_z.rows() );
	VecX solver_termB = VecX::Zero( dof );

	// Run a timestep
	int s_i = 0;
	for( ; s_i < m_settings.admm_iters; ++s_i ){

		if( m_settings.record_obj ){
			VecX Msqrt = m_masses.cwiseSqrt();
			VecX resid = 0.5 * (1.0/dt) * Msqrt.asDiagonal() * (curr_x - x_bar);
			double fx = resid.squaredNorm();
			double gx = 0.0;
			for( int i=0; i<n_energyterms; ++i ){ gx += energyterms[i]->energy( m_D, curr_x ); }
			m_runtime.f.emplace_back( fx );
			m_runtime.g.emplace_back( gx );
		} // end record obj

		// Local step
		t.reset();
		#pragma omp parallel for num_threads(n_threads)
		for( int i=0; i<n_energyterms; ++i ){
			energyterms[i]->update( m_D, curr_x, curr_z, curr_u );
		}
		m_runtime.local_ms += t.elapsed_ms();

		// Global step
		t.reset();
		solver_termB.noalias() = M_xbar + solver_Dt_Wt_W * ( curr_z - curr_u );
		m_runtime.inner_iters += m_linsolver->solve( curr_x, solver_termB );
		m_runtime.global_ms += t.elapsed_ms();

	} // end solver loop

	// Computing new velocity and setting the new state
	m_v.noalias() = ( curr_x - m_x ) * ( 1.0 / dt );
	m_x = curr_x;

	// Output run time
	if( m_settings.verbose > 0 ){ m_runtime.print(m_settings); }
} // end timestep iteration


void Solver::set_pins( const std::vector<int> &inds, const std::vector<Vec3> &points ){

	const int dof = m_x.rows();
	bool pin_in_place = points.size() != inds.size();
	if( (dof == 0 && pin_in_place) || (pin_in_place && points.size() > 0) ){
		throw std::runtime_error("**Solver::set_pins Error: Bad input.");
	}

	pins.clear();
	int n_pins = inds.size();
	for( int i=0; i<n_pins; ++i ){
		int idx = inds[i];
		if( pin_in_place ){
			pins[idx] = m_x.segment<3>(idx*3);
		} else {
			pins[idx] = points[i];
		}
	}
}

void Solver::add_obstacle( std::shared_ptr<PassiveCollision> obj ){
	m_collider->add_passive_obj(obj);
}

bool Solver::initialize( const Settings &settings_ ){
	using namespace Eigen;
	m_settings = settings_;

	mcl::MicroTimer t;
	const int dof = m_x.rows();
	if( m_settings.verbose > 0 ){ std::cout << "Solver::initialize: " << std::endl; }

	if( m_settings.timestep_s <= 0.0 ){
		std::cerr << "\n**Solver Error: timestep set to " << m_settings.timestep_s <<
			"s, changing to 1/24s." << std::endl;
		m_settings.timestep_s = 1.0/24.0;
	}
	if( !( m_masses.rows()==dof && dof>=3 ) ){
		std::cerr << "\n**Solver Error: Problem with node data!" << std::endl;
		return false;
	}
	if( m_v.rows() != dof ){ m_v.resize(dof); }

	// Clear previous runtime stuff settings
	m_v.setZero();

	// Set up the selector matrix (D) and weight (W) matrix
	std::vector<Eigen::Triplet<double> > triplets;
	std::vector<double> weights;
	int n_energyterms = energyterms.size();
	for(int i = 0; i < n_energyterms; ++i){
		energyterms[i]->get_reduction( triplets, weights );
	}
	m_W_diag = Eigen::Map<VecX>(&weights[0], weights.size());
	int n_D_rows = weights.size();
	m_D.resize( n_D_rows, dof );
	m_D.setZero();
	m_D.setFromTriplets( triplets.begin(), triplets.end() );
	m_Dt = m_D.transpose();

	// Compute mass matrix
	SparseMat M( dof, dof );
	Eigen::VectorXi nnz = Eigen::VectorXi::Ones( dof ); // non zeros per column
	M.reserve(nnz);
	for( int i=0; i<dof; ++i ){ M.coeffRef(i,i) = m_masses[i]; }

	// Set global matrices
	SparseMat W( n_D_rows, n_D_rows );
	W.reserve(n_D_rows);
	for( int i=0; i<n_D_rows; ++i ){ W.coeffRef(i,i) = m_W_diag[i]; }
	const double dt2 = (m_settings.timestep_s*m_settings.timestep_s);
	solver_Dt_Wt_W = dt2 * m_Dt * W * W;
	solver_termA = M + SparseMat(solver_Dt_Wt_W * m_D);

	// Set up the linear solver
	switch (m_settings.linsolver){
		default: {
			
		} break;
		case 1: {

		} break;
	}

	// If we haven't set a global solver, make one:
	if( !m_linsolver ){ throw std::runtime_error("What happened to the global solver?"); }
	m_linsolver->update_system( solver_termA );

	// All done
	if( m_settings.verbose >= 1 ){ printf("%d nodes, %d energy terms\n", (int)m_x.size()/3, (int)energyterms.size() ); }
	return true;

} // end init


template<typename T> void myclamp( T &val, T min, T max ){ if( val < min ){ val = min; } if( val > max ){ val = max; } }
bool Solver::Settings::parse_args( int argc, char **argv ){

	// Check args with params
	for( int i=1; i<argc-1; ++i ){
		std::string arg( argv[i] );
		std::stringstream val( argv[i+1] );
		if( arg == "-help" || arg == "--help" || arg == "-h" ){ help(); return true; }
		else if( arg == "-dt" ){ val >> timestep_s; }
		else if( arg == "-v" ){ val >> verbose; }	
		else if( arg == "-it" ){ val >> admm_iters; }
		else if( arg == "-g" ){ val >> gravity; }
		else if( arg == "-r" ){ record_obj = true; }
	}

	// Check if last arg is one of our no-param args
	std::string arg( argv[argc-1] );
	if( arg == "-help" || arg == "--help" || arg == "-h" ){ help(); return true; }
	else if( arg == "-r" ){ record_obj = true; }

	return false;

} // end parse settings args

void Solver::Settings::help(){
	std::stringstream ss;
	ss << "\n==========================================\nArgs:\n" <<
		"\t-dt: time step (s)\n" <<
		"\t-v: verbosity (higher -> show more)\n" <<
		"\t-it: # admm iters\n" <<
		"\t-g: gravity (m/s^2)\n" <<
		"\t-r: record objective value \n" <<
	"==========================================\n";
	printf( "%s", ss.str().c_str() );
}

void Solver::RuntimeData::print( const Settings &settings ){
	std::cout << "\nTotal global step: " << global_ms << "ms";;
	std::cout << "\nTotal local step: " << local_ms << "ms";
	std::cout << "\nAvg global step: " << global_ms/double(settings.admm_iters) << "ms";;
	std::cout << "\nAvg local step: " << local_ms/double(settings.admm_iters) << "ms";
	std::cout << "\nADMM Iters: " << settings.admm_iters;
	std::cout << "\nAvg Inner Iters: " << float(inner_iters) / float(settings.admm_iters);
	std::cout << std::endl;
}

