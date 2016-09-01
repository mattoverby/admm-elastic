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
friend class System;
public:
	Force() : weight(0.f) { Di.setZero(); }
	virtual ~Force() {}

	virtual void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep ) = 0;
	virtual void computeDi( int dof ) = 0; // called AFTER initialize!
	virtual void update( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const = 0;

	int global_idx; // Global index is the position of this force in the global u and z vectors
	double weight; // Weight is computed BY the force based on stiffness

protected:
	// Di should only be manipulated through these function calls (setDi and getDi).
	// This is a little strange but affords some protection when initializing the system.
	const Eigen::SparseMatrix<double> *getDi() const { assert(Di.nonZeros()>0); return &Di; }
	void setDi( Eigen::SparseMatrix<double> &Di_ ){ Di = Di_; }

	// Called by System::initialize:
	virtual void set_global_idx( int idx ){ global_idx = idx; }

	// Return the energy of a force. This is used for convergence tests
//	virtual double getEnergy( const Eigen::VectorXd& Dx, Eigen::VectorXd& grad ) const {
//		std::cerr << "**Error: getEnergy not implemented" << std::endl; exit(0);
//	}

private:
	// Di should only be manipulated through the function calls (setDi and getDi).
	// TODO not store personal Di to save memory. Currently it's convenient.
	Eigen::SparseMatrix<double> Di;

}; // end class force


//
//	Spring Force
//
class Spring : public Force {
public:
	Spring( int idx0_, int idx1_, double stiffness_ ) : idx0(idx0_), idx1(idx1_), stiffness(stiffness_) {}
	void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep );
	void computeDi( int dof );
	void update( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;
//	double getEnergy( const Eigen::VectorXd& Dx, Eigen::VectorXd& grad ) const;

	int idx0, idx1;
	double stiffness, rest_length;

protected:

}; // end class spring


} // end namespace admm

#endif




