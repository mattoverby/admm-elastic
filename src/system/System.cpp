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


bool System::step( int solver_iters ){

	// Loop the step callbacks
	for( int cb_i=0; cb_i<step_callbacks.size(); ++cb_i ){ step_callbacks[cb_i](this); }

	const int n_forces = forces.size();

	// Initialize ADMM vars
	// curr_u.setZero(); // Let curr_u be its values at last timestep (better convergence)
	curr_z = m_D*m_x;

	// Take an explicit step to get predicted node positions
	// with simple forces (e.g. wind/gravity).
	// These are parallelized internally.
	for( int i=0; i<explicit_forces.size(); ++i ){
		explicit_forces[i]->update( timestep_s, m_x, m_v, m_masses );
	}

	// Position without constraints
	VectorXd x_bar = m_x + timestep_s * m_v;
	VectorXd M_xbar = m_masses.asDiagonal() * x_bar;
	VectorXd curr_x = x_bar; // Temperorary x used in optimization

	// Run a timestep
	for( int s_i=0; s_i < solver_iters; ++s_i ){

		if( m_verbose >= 2 ){ print_progress(s_i,solver_iters); }

		// Do the matrix multiply here instead of per-force, and then just pass Dx.
		Dx = m_D*curr_x;

		// Local step (uses curr_x, and does zi and ui updates on each force)
		#pragma omp parallel for
		for( int i = 0; i < n_forces; ++i ){ forces[i]->update(timestep_s,Dx,curr_u,curr_z); }

		// Global step (sets curr_x)
		solver_termB = M_xbar + solver_dt2_Dt_Wt_W * ( curr_z - curr_u );
		curr_x = solver.solve( solver_termB );

		// You can test for convergence and early exit by computing residuals (Eq. 22, 23):
		// r = W*(Dx-curr_z), s = Dt*Wt*W*(curr_z-last_z)

	} // end solver loop

	// Computing new velocity and setting the new state
	m_v = ( curr_x - m_x ) * ( 1.0 / timestep_s );
	m_x = curr_x;
	elapsed_s += timestep_s;

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


bool System::initialize( double timestep_ ){

	// The order of global values being initialized is a little strange,
	// along with the getting/setting of local force variables. This
	// is done primarily for performance reasons.

	if( m_verbose >= 1 ){ std::cout << "Solver::initialize: " << std::flush; }

	timestep_s = timestep_;
	assert(timestep_s>0.0);
	if( !( m_masses.size()==m_x.size() && m_x.size()==m_v.size() && m_x.size()>=3 ) ){
		std::cerr << "\n**Solver Error: Problem with node data!" << std::endl;
		return false;
	}
	m_v.setZero();

	// Make sure we aren't already initialized
	if( initialized ){ std::cerr << "\n**Solver Error: Already initialized!" << std::endl; return false; }
	const int dof = m_x.size();

	// Compute local Di matrices in parallel.
	// In the future we should avoid doing storing Di matrices on forces to
	// reduce memory consumption.
	#pragma omp parallel for
	for(int i = 0; i < forces.size(); ++i){
		forces[i]->initialize( m_x, m_v, m_masses, timestep_s );
		forces[i]->computeDi( dof );
	}

	// Set indices into the global buffers
	int zu_idx=0;
	for(int i = 0; i < forces.size(); ++i){
		forces[i]->set_global_idx( zu_idx );
		int Di_rows = forces[i]->getDi()->rows();
		zu_idx += Di_rows;
	}

	int D_numRows = zu_idx;
	int D_numCols = m_x.size();
	int n_nodes = D_numCols/3;

	// Building the global D and W matrices
	// TODO: From triplets and combine with loop above instead of dense matrices
	m_W_diag.resize( D_numRows );
	Eigen::MatrixXd tempD( D_numRows, D_numCols ); tempD.setZero();
	int curr_row = 0;
	for(int i = 0; i < forces.size(); ++i){
		int numDiRows = forces[i]->getDi()->rows();
		tempD.block( curr_row, 0, numDiRows, D_numCols ) = *forces[i]->getDi();
		for( int j=0; j<numDiRows; ++j ){ m_W_diag[ curr_row + j ] = forces[i]->weight; }
		curr_row += numDiRows;
	}

	// Use transpose of D as well
	m_D = tempD.sparseView();
	SparseMatrix<double> Dt = m_D.transpose();

	// Compute mass matrix
	Eigen::SparseMatrix<double> M( dof, dof );
	Eigen::VectorXi nnz = Eigen::VectorXi::Ones( dof ); // non zeros per column
	M.reserve(nnz);
	for( int i=0; i<dof; ++i ){ M.coeffRef(i,i) = m_masses[i]; }

	// Setup the solver
	DiagonalMatrix<double,Dynamic> W = m_W_diag.asDiagonal();
	SparseMatrix<double> solver_termA = ( M + timestep_s*timestep_s * Dt * W * W * m_D );
	solver_dt2_Dt_Wt_W = timestep_s*timestep_s * Dt * W * W;
	solver.compute( solver_termA );

	// Allocate space for our ADMM vars
	solver_termB.resize( m_D.rows() );
	Dx.resize( m_D.rows() );
	curr_u.resize( m_D.rows() );
	curr_u.setZero();
	curr_z.resize( m_D.rows() );

	if( m_verbose >= 1 ){
		std::cout <<  m_x.size()/3 << " nodes, " << forces.size() << " forces" << std::endl;
	}

	initialized = true;
	return true;

} // end init


void System::recompute_weights(){

	// Update the weight matrix
	#pragma omp parallel for
	for(int i = 0; i < forces.size(); ++i){
		int Di_rows = forces[i]->getDi()->rows();
		Eigen::VectorXd weights = Eigen::VectorXd::Ones(Di_rows)*forces[i]->weight;
		m_W_diag.segment( forces[i]->global_idx, Di_rows ) = weights;
	}

	// Setup the solver
	int dof = m_masses.size();
	Eigen::SparseMatrix<double> M( dof, dof ); // needed because eigen doesn't like diagonal*sparse
	Eigen::VectorXi nnz = Eigen::VectorXi::Ones( dof ); // non zeros per column
	M.reserve(nnz);
	for( int i=0; i<dof; ++i ){ M.coeffRef(i,i) = m_masses[i]; }
	SparseMatrix<double> Dt = m_D.transpose();
	DiagonalMatrix<double,Dynamic> W = m_W_diag.asDiagonal();
	solver_dt2_Dt_Wt_W = timestep_s*timestep_s * Dt * W * W;
	SparseMatrix<double> solver_termA = ( M + solver_dt2_Dt_Wt_W * m_D );
	solver.compute( solver_termA );

}


void System::print_progress( int iter, int max_iters ){
	double progress = double(iter)/double(max_iters) * 100.0;
	printf( "\r %.3f %%", progress );
}


