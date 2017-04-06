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

#ifndef ADMM_FORCE_H
#define ADMM_FORCE_H 1

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <memory>
#include "cppoptlib/solver/lbfgssolver.h"

namespace admm {

//
//	Force base class
//
//	Weight should be calculated in the initialize function, then retrieved by the system
//	in System::initialize. For certain forces, optimized w_i is known.
//
class Force {
public:
	int global_idx; // Global index is the position of this force in the global u and z vectors
	double weight; // Weight is computed BY the force based on stiffness

	Force() : weight(0.f) {}
	virtual ~Force() {}

	// Called in System::initialize to compute local variables
	virtual void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep ){}

	// Get triplets for the selector (D) matrix, called after initialize
	virtual void get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights ) = 0;

	// Called in System::step
	virtual void project( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const = 0;

	// Set an epsilon for collision/sliding/etc...
	virtual void set_eps( double eps ){}

}; // end class force


//
//	Spring Force
//
class Spring : public Force {
public:
	Spring( int idx0_, int idx1_, double stiffness_ ) : idx0(idx0_), idx1(idx1_), stiffness(stiffness_) {}
	void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep );
	void get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights );
	void project( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;
	int idx0, idx1;
	double stiffness, rest_length;

}; // end class spring


} // end namespace admm

#endif




