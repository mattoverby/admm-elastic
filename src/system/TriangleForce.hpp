// Copyright (c) 2016, University of Minnesota
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

#ifndef ADMM_TRIANGLEFORCE_H
#define ADMM_TRIANGLEFORCE_H 1

#include "Force.hpp"

namespace admm {

//
//	LimitedTriangleStrain
//
class LimitedTriangleStrain : public Force {
public:
	LimitedTriangleStrain( int id0_, int id1_, int id2_, double stiffness_, double limit_min_, double limit_max_ ) :
		id0(id0_), id1(id1_), id2(id2_), stiffness(stiffness_), limit_min(limit_min_), limit_max(limit_max_) {}
	void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep );
	void computeDi( int dof );
	void update( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;

	int id0, id1, id2;
	double stiffness, limit_min, limit_max;
	Eigen::Matrix<double,3,2> B;
	double area;

}; // end class limited triangle strain



//
//	PDTriangleStrain (projective dynamics style)
//
class PDTriangleStrain : public Force {
public:
	PDTriangleStrain( int id0_, int id1_, int id2_, double stiffness_, double limit_min_, double limit_max_ ) :
		id0(id0_), id1(id1_), id2(id2_), stiffness(stiffness_), limit_min(limit_min_), limit_max(limit_max_) {}
	void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep );
	void computeDi( int dof );
	void update( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;

	int id0, id1, id2;
	double stiffness, limit_min, limit_max;
	Eigen::Matrix<double,3,2> B;
	double area;

}; // end class limited triangle strain


} // end namespace admm

#endif




