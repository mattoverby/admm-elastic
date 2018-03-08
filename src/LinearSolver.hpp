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

#ifndef ADMM_LINEARSOLVER_H
#define ADMM_LINEARSOLVER_H

#include "SolverLog.hpp"
#include <Eigen/SparseCholesky>
#include <unordered_map>
#include <memory>
#ifdef EIGEN_USE_MKL_VML
#include <Eigen/PardisoSupport>
#endif

namespace admm {

//
//	Linear solver base class
//
class LinearSolver {
public:
	typedef Eigen::Matrix<double,3,1> Vec3;
	typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;
	typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SparseMat;

	SolverLog logger;

	// Updates solver data
	virtual void update_system( const SparseMat &A_ ) = 0;

	// Solve Ax=b subject to constraints
	virtual int solve( VecX &x, const VecX &b ) = 0;

	// Helper class for avoiding divide-by-zero
	static bool is_zero(double x){ return std::abs(x)<std::numeric_limits<double>::min(); }

}; // end class linear solver


//
//	Classic LDLT solver that does not handle constraints (SCA 2016)
//
class LDLTSolver : public LinearSolver {
public:
	typedef Eigen::Matrix<double,3,1> Vec3;
	typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;
	typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SparseMat;

	#ifdef EIGEN_USE_MKL_VML
	typedef Eigen::PardisoLDLT< Eigen::SparseMatrix<double> > Cholesky;
	#else
	typedef Eigen::SimplicialLDLT< Eigen::SparseMatrix<double> > Cholesky;
	#endif

	SparseMat A;
	std::unique_ptr<Cholesky> m_cholesky;

	LDLTSolver(){
		m_cholesky = std::unique_ptr<Cholesky>( new Cholesky() );
	}

	// Does a cholesky factorization on the system matrix
	void update_system( const SparseMat &A_ ){
		int dim = A_.rows();
		if( dim != A_.cols() || dim == 0 ){ throw std::runtime_error("**LDLTSolver Error: Bad dimensions in A"); }
		A = A_;
		m_cholesky->compute(A);
	}

	// Solves for x given A, linear constraints (C), and pinning subspace (P)
	int solve( VecX &x, const VecX &b0 ){
		x = m_cholesky->solve(b0);
		return 1;
	}

}; // end class linear solver

} // ns admm

#endif
