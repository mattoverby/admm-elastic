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

#ifndef ADMM_UZAWACGLINEARSOLVER_H
#define ADMM_UZAWACGLINEARSOLVER_H

#include "LinearSolver.hpp"
#include "ConstraintSet.hpp"

namespace admm {

//
// Uzawa Solver for saddle point system
// [ A  C* ] [ x ] = [ b ]
// [ C  0  ] [ y ] = [ c ]
//
class UzawaCG : public LinearSolver {
public:
	typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;
	typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SparseMat;
	typedef Eigen::SimplicialLDLT< Eigen::SparseMatrix<double> > Cholesky;

	int max_iters;
	double m_tol;
	std::unique_ptr<Cholesky> cholesky;
	std::shared_ptr<ConstraintSet> constraints;

	UzawaCG( std::shared_ptr<ConstraintSet> constraints_ ) :
		max_iters(20), m_tol(1e-10), constraints(constraints_) {
		cholesky = std::unique_ptr<Cholesky>( new Cholesky() );
	}

	UzawaCG() : UzawaCG(std::make_shared<ConstraintSet>(ConstraintSet())) {}

	void update_system( const SparseMat &A_ ){
		A = A_;
		cholesky->compute(A);
	}

	// Solve the linear system
	int solve( VecX &x, const VecX &b0 ){
		using namespace Eigen;
		logger.reset();

		// Make constraint matrix if needed
		int dof = A.cols();
		constraints->make_matrix(dof,true,true);
		const SparseMat &C = constraints->m_C;
		const SparseMat &Ct = constraints->m_Ct;
		const VecX &c = constraints->m_c;
		if( y.rows() != C.rows() ){ y = VecX::Zero(c.rows()); }

		// If there are no constraints, just use the
		// prefactored linear solve.
		if( C.nonZeros()==0 ){
			x = cholesky->solve(b0);
			return 1;
		}

		VecX q1 = b0 - Ct*y;
		x = cholesky->solve( q1 );

		VecX r = C*x - c;
		VecX d = r;
		VecX q2, q3;
		double tol2 = m_tol*m_tol;

		int iter = 0;
		for( ; iter<max_iters; ++iter ){

			q1 = Ct*d;
			q2 = cholesky->solve( q1 );
			q3 = C*q2;

			double denom = d.dot(q3);
			if( is_zero(denom) ){ break; }
			double alpha = d.dot(r) / denom;

			// Take a step
			x -= alpha * q2;
			y += alpha * d;
			r -= alpha * q3;

			// Exit if resid is low enough
			logger.add(x);
			if( r.squaredNorm() < tol2 ){ break; }

			denom = d.dot(q3);
			if( is_zero(denom) ){ break; }
			double beta = r.dot(q3) / denom;
			d = r - ( beta * d );

		} // end cg iters

		logger.finalize(A,x,b0);
		return iter;

	} // end uzawa solve

private:
	VecX y; // lagrange mults
	SparseMat A;

};

} // ns admm

#endif
