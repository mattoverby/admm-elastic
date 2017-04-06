// Copyright (c) 2016, University of Minnesota
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
using namespace Eigen;


bool System::step(){

	// Loop the step callbacks
	for( int cb_i=0; cb_i<pre_step_callbacks.size(); ++cb_i ){ pre_step_callbacks[cb_i](this); }

	const int n_forces = forces.size();
	const double dt = settings.timestep_s;

	// Take an explicit step to get predicted node positions
	// with simple forces (e.g. wind/gravity).
	// These are parallelized internally.
	for( int i=0; i<explicit_forces.size(); ++i ){
		explicit_forces[i]->project( dt, m_x, m_v, m_masses );
	}

	// Initialize ADMM vars
	// curr_u.setZero(); // Let curr_u be its values at last timestep (better convergence)
	curr_z.noalias() = m_D*m_x;

	// Position without constraints
	VectorXd x_bar = m_x + dt * m_v;
	VectorXd M_xbar = m_masses.asDiagonal() * x_bar;
	VectorXd curr_x = x_bar; // Temperorary x used in optimization

	// Run a timestep
	for( int s_i=0; s_i < settings.admm_iters; ++s_i ){

		// Do the matrix multiply here instead of per-force, and then just pass Dx.
		Dx = m_D*curr_x;

		// Local step (uses curr_x, and does zi and ui updates on each force)
#pragma omp parallel for
		for( int i = 0; i < n_forces; ++i ){ forces[i]->project(dt,Dx,curr_u,curr_z); }

		// Global step (sets curr_x)
		solver_termB.noalias() = M_xbar + solver_dt2_Dt_Wt_W * ( curr_z - curr_u );
		curr_x = solver.solve( solver_termB );

		// You can test for convergence and early exit by computing residuals (Eq. 22, 23):
		// r = W*(Dx-curr_z), s = Dt*Wt*W*(curr_z-last_z)

	} // end solver loop

	// Computing new velocity and setting the new state
	m_v.noalias() = ( curr_x - m_x ) * ( 1.0 / dt );
	m_x = curr_x;
	elapsed_s += dt;

	return true;
}


int System::add_nodes( Eigen::VectorXd x, Eigen::VectorXd m ){

	int old_system_nodes = m_x.size();
	int new_system_nodes = x.size();

	m_x.conservativeResize(old_system_nodes+new_system_nodes);
	m_v.conservativeResize(old_system_nodes+new_system_nodes);
	m_masses.conservativeResize(old_system_nodes+new_system_nodes);

	for( int i=0; i<x.size(); ++i ){
		int idx = old_system_nodes+i;
		m_x[idx] = x[i];
		m_v[idx] = 0.0;
		m_masses[idx] = m[i];
	}

	return (old_system_nodes+new_system_nodes)/3;
}


bool System::initialize(){

	const int dof = m_x.size();
	if( settings.verbose > 0 ){ std::cout << "Solver::initialize: " << std::endl; }

	if( settings.timestep_s <= 0.0 ){
		std::cerr << "\n**Solver Error: timestep set to " << settings.timestep_s <<
			"s, changing to 0.04s." << std::endl;
		settings.timestep_s = 0.04;
	}
	if( !( m_masses.size()==m_x.size() && m_x.size()>=3 ) ){
		std::cerr << "\n**Solver Error: Problem with node data!" << std::endl;
		return false;
	}
	if( m_v.size() < m_x.size() ){ m_v.resize(m_x.size()); }
	m_v.setZero();

	// Initialize forces
#pragma omp parallel for
	for(int i = 0; i < forces.size(); ++i){
		forces[i]->initialize( m_x, m_v, m_masses, settings.timestep_s );
	}

	// Set up the selector matrix (D) and weight (W) matrix
	std::vector<Eigen::Triplet<double> > triplets;
	std::vector<double> weights;
	for(int i = 0; i < forces.size(); ++i){ forces[i]->get_selector( m_x, triplets, weights ); }
	m_W_diag = Eigen::Map<Eigen::VectorXd>(&weights[0], weights.size());
	m_D.resize( weights.size(), dof );
	m_D.setFromTriplets( triplets.begin(), triplets.end() );
	SparseMatrix<double> Dt = m_D.transpose();

	// Compute mass matrix
	Eigen::SparseMatrix<double> M( dof, dof );
	Eigen::VectorXi nnz = Eigen::VectorXi::Ones( dof ); // non zeros per column
	M.reserve(nnz);
	for( int i=0; i<dof; ++i ){ M.coeffRef(i,i) = m_masses[i]; }

	// Setup the solver
	DiagonalMatrix<double,Dynamic> W = m_W_diag.asDiagonal();
	Eigen::SparseMatrix<double> solver_termA = ( M + settings.timestep_s*settings.timestep_s * Dt * W * W * m_D );
	solver_dt2_Dt_Wt_W = settings.timestep_s*settings.timestep_s * Dt * W * W;
	solver.compute( solver_termA );

	// Allocate space for our ADMM vars
	solver_termB.resize( m_D.rows() );
	Dx.resize( m_D.rows() );
	curr_u.resize( m_D.rows() );
	curr_u.setZero();
	curr_z.resize( m_D.rows() );

	if( settings.verbose >= 1 ){
		std::cout <<  m_x.size()/3 << " nodes, " << forces.size() << " forces" << std::endl;
	}

	initialized = true;
	return true;

} // end init


void System::recompute_weights(){

	int dof = m_masses.size();

	// Update the weight matrix
	std::vector<Eigen::Triplet<double> > triplets;
	std::vector<double> weights;
	for(int i = 0; i < forces.size(); ++i){
		forces[i]->get_selector( m_x, triplets, weights );
	}
	m_W_diag = Eigen::Map<Eigen::VectorXd>(&weights[0], weights.size());

	// Setup the solver
	Eigen::SparseMatrix<double> M( dof, dof ); // needed because eigen doesn't like diagonal*sparse
	Eigen::VectorXi nnz = Eigen::VectorXi::Ones( dof ); // non zeros per column
	M.reserve(nnz); for( int i=0; i<dof; ++i ){ M.coeffRef(i,i) = m_masses[i]; }
	DiagonalMatrix<double,Dynamic> W = m_W_diag.asDiagonal();
	solver_dt2_Dt_Wt_W = settings.timestep_s*settings.timestep_s * m_D.transpose() * W * W;
	SparseMatrix<double> solver_termA = ( M + solver_dt2_Dt_Wt_W * m_D );
	solver.compute( solver_termA );
}


void System::Settings::parse_args( int argc, char **argv ){

	// Check args with params
	for( int i=1; i<argc-1; ++i ){
		std::string arg( argv[i] );
		std::stringstream val( argv[i+1] );
		if( arg == "-help" ){ help(); }
		else if( arg == "-dt" ){ val >> timestep_s; }
		else if( arg == "-v" ){ val >> verbose; }	
		else if( arg == "-it" ){ val >> admm_iters; }
	}

	// Check if last arg is one of our no-param args
	std::string arg( argv[argc-1] );
	if( arg == "-help" ){ help(); }

} // end parse settings args

void System::Settings::help(){
	std::stringstream ss;
	ss << "\n==========================================\nArgs:\n" <<
		"\t-dt: time step (s)\n" <<
		"\t-v: verbosity (higher -> show more)\n" <<
		"\t-it: # admm iters\n" <<
	"==========================================\n";
	printf( "%s", ss.str().c_str() );
}


