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
#include "UzawaCG.hpp"
#include <unordered_map>

using namespace admm;
using namespace Eigen;

Solver::Solver() : initialized(false) {
	m_constraints = std::make_shared<ConstraintSet>( ConstraintSet() );
}

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
	bool detect_passive = m_settings.linsolver!=1;
	bool penalty_collisions = m_constraints->constraint_w > 1.0;
	m_constraints->collider->clear_hits();
//	m_constraints->collider->detect( curr_x, detect_passive );
//	m_constraints->matrix_needs_update = true;

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

		if( !penalty_collisions ){ m_constraints->collider->clear_hits(); }
		m_constraints->collider->detect( surface_inds, curr_x, detect_passive );
		m_constraints->matrix_needs_update = true;

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

	int n_pins = inds.size();
	const int dof = m_x.rows();
	bool pin_in_place = (int)points.size() != n_pins;
	if( (dof == 0 && pin_in_place) || (pin_in_place && points.size() > 0) ){
		throw std::runtime_error("**Solver::set_pins Error: Bad input.");
	}

	m_constraints->matrix_needs_update = true;
	m_constraints->pins.clear();
	for( int i=0; i<n_pins; ++i ){
		int idx = inds[i];
		if( pin_in_place ){
			m_constraints->pins[idx] = m_x.segment<3>(idx*3);
		} else {
			m_constraints->pins[idx] = points[i];
		}
	}

	// If we're using energy based hard constraints, the pin locations may change
	// but which vertex is pinned may NOT change (aside from setting them to
	// active or inactive). So we need to do some extra work here.
	if( initialized && (m_settings.linsolver==0 || m_settings.linsolver==2) ){

		// Set all pins inactive, then update to active if they are set
		std::unordered_map<int, std::shared_ptr<SpringPin> >::iterator pIter = m_pin_energies.begin();
		for( ; pIter != m_pin_energies.end(); ++pIter ){
			pIter->second->set_active(false);
		}

		// Update pin locations/active
		for( int i=0; i<n_pins; ++i ){
			int idx = inds[i];
			pIter = m_pin_energies.find(idx);
			if( pIter == m_pin_energies.end() ){
				std::stringstream err;
				err << "**Solver::set_pins Error: Constraint for " << idx << " not found.\n";
				throw std::runtime_error(err.str().c_str());
			}
			pIter->second->set_active(true);
			pIter->second->set_pin( m_constraints->pins[idx] );
		}

	} // end set energy-based pins
}

void Solver::add_obstacle( std::shared_ptr<PassiveCollision> obj ){
	m_constraints->collider->add_passive_obj(obj);
}

void Solver::add_dynamic_collider( std::shared_ptr<DynamicCollision> obj ){
	m_constraints->collider->add_dynamic_obj(obj);
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

	// If we want energy-based constraints, set them up now.
	if( m_settings.linsolver==0 || m_settings.linsolver==2){
		std::unordered_map<int,Vec3>::iterator pinIter = m_constraints->pins.begin();
		for( ; pinIter != m_constraints->pins.end(); ++pinIter ){
			m_pin_energies[ pinIter->first ] = std::make_shared<SpringPin>( SpringPin(pinIter->first,pinIter->second) );
			energyterms.emplace_back( m_pin_energies[ pinIter->first ] );			
		}
	} // end create energy based hard constraints

	// Set up the selector matrix (D) and weight (W) matrix
	std::vector<Eigen::Triplet<double> > triplets;
	std::vector<double> weights;
	int n_energyterms = energyterms.size();
	for(int i = 0; i < n_energyterms; ++i){
		energyterms[i]->get_reduction( triplets, weights );
	}

	// Create the Selector+Reduction matrix
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
			m_linsolver = std::make_shared<LDLTSolver>( LDLTSolver() );
		} break;
		case 1: {
			m_linsolver = std::make_shared<NodalMultiColorGS>( NodalMultiColorGS(m_constraints) );
			m_constraints->constraint_w = std::sqrt(m_W_diag.maxCoeff()*3.0);
		} break;
		case 2: {
			m_linsolver = std::make_shared<UzawaCG>( UzawaCG(m_constraints) );
			m_constraints->constraint_w = 1.0;
			// Uzawa solver needs odd number of iterations due to the way collisions
			// are handled. Specifically, they are fully resolved on even iterations, thus
			// not detected on the odd ones and penetration may occur at the end
			// of the timestep. A quick hack for now is to make sure we have odd iters...
			if( m_settings.admm_iters % 2 == 0 ){
				m_settings.admm_iters++;
			}
		} break;
	}

	// If we haven't set a global solver, make one:
	if( !m_linsolver ){ throw std::runtime_error("What happened to the global solver?"); }
	if( m_settings.constraint_w > 0.0 ){ m_constraints->constraint_w = m_settings.constraint_w; }
	m_linsolver->update_system( solver_termA );

	// Make sure they don't have any collision obstacles
	if( m_settings.linsolver==0 ){
		if( m_constraints->collider->passive_objs.size() > 0 ||
			m_constraints->collider->dynamic_objs.size() > 0 ){
			throw std::runtime_error("**Solver::add_obstacle Error: No collisions with LDLT solver");
		}
	}

	// All done
	if( m_settings.verbose >= 1 ){ printf("%d nodes, %d energy terms\n", (int)m_x.size()/3, (int)energyterms.size() ); }
	initialized = true;
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
		else if( arg == "-ls" ){ val >> linsolver; }
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
		"\t-ls: linear solver (0=LDLT, 1=NCMCGS) \n" <<
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

