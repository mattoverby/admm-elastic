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

#ifndef ADMM_SOLVERLOG_H
#define ADMM_SOLVERLOG_H

#include <Eigen/Sparse>
#include "MCL/MicroTimer.hpp"

namespace admm {

class SolverLog {
public:
	typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;
	typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SparseMat;

	SolverLog(){ x_star=VecX::Zero(1); }

	void reset(){
		errors.clear();
		runtimes.clear();
		t.reset();
	}

	void add( const VecX &x ){
		if(skip(x)){ return; }
		if( errors.size()==0 ){
			runtimes.push_back( 0.0 );
			t.reset();
		}
		else{ runtimes.push_back( t.elapsed_ms() ); }
		if( errors.size() == 0 ){ x0 = x; }
		double numer = ( x_star - x ).norm();
		double denom = ( x_star - x0 ).norm();
		errors.push_back( numer / denom );
	}

	void finalize( const SparseMat &A, const VecX &x, const VecX &b ){
		if( skip(x) ){ return; }
		final_r = ( A*x - b ).norm();
	}

	std::vector<double> errors; // per iteration residuals
	std::vector<double> runtimes; // per iteration run time
	double final_r;// final ||Ax-b||
	VecX x_star; // true solution, input

private:
	inline bool skip( const VecX &x ){ return x_star.rows()!=x.rows(); }
	VecX x0; // set in add
	mcl::MicroTimer t;
};

}

#endif
