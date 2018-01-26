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

#ifndef ADMM_EXPLICITFORCE_H
#define ADMM_EXPLICITFORCE_H 1

#include <Eigen/Dense>
#include <vector>

namespace admm {


//
//	Explicit Forces are a bit different in that they are applied before
//	optimization (computing xbar) to everything in the domain.
//
class ExplicitForce {
public:
	virtual void project( double dt, Eigen::VectorXd &x, Eigen::VectorXd &v, Eigen::VectorXd &m ) const = 0;
};


class WindForce : public ExplicitForce {
public:
	// Input is a list of all triangles the wind force affects.
	WindForce( std::vector< int > &tris_ ) : tris(tris_) { direction=Eigen::Vector3d(0,0,0); }
	void project( double dt, Eigen::VectorXd &x, Eigen::VectorXd &v, Eigen::VectorXd &m ) const;
	std::vector< int > tris;
	Eigen::Vector3d direction;
};


} // end namespace admm

#endif




