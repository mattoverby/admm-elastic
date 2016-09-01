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

#ifndef ADMM_ANCHORFORCE_H
#define ADMM_ANCHORFORCE_H 1

#include "Force.hpp"

namespace admm {

namespace helper {

	// Useful for moving control points smoothly
	// total_elapsed_dt = system time elapsed sum
	// start_dt = time to start the movement
	// move_dt = time it takes to move from start position to end position
	static inline Eigen::Vector3d smooth_move( double total_elapsed_dt, double start_dt, double end_dt, Eigen::Vector3d start, Eigen::Vector3d end ){
		if( total_elapsed_dt < start_dt ){ return start; }
		double tRatio = (total_elapsed_dt-start_dt) / (end_dt-start_dt);
		if( tRatio > 1.0 ){ return end; }
		Eigen::Vector3d displacement = end - start;
		return ( start + (3.0*tRatio*tRatio - 2.0*tRatio*tRatio*tRatio)*displacement );
	}

	static inline Eigen::Vector3d linear_move( double total_elapsed_dt, double start_dt, double end_dt, Eigen::Vector3d start, Eigen::Vector3d end ){
		if( total_elapsed_dt < start_dt ){ return start; }
		double tRatio = (total_elapsed_dt-start_dt) / (end_dt-start_dt);
		if( tRatio > 1.0 ){ return end; }
		Eigen::Vector3d displacement = end - start;
		return ( start + displacement );
	}

} // end namespace helper


//
//	StaticAnchor force
//
class StaticAnchor : public Force {
public:
	StaticAnchor( int idx_ ) : idx(idx_) {}
	void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep );
	void computeDi( int dof );
	void update( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;

	int idx;
	Eigen::Vector3d pos;

protected:

}; // end class force


//
//	MovingAnchor 
//	Bind an anchor to a control point
//
class MovingAnchor; // forward declaration
class ControlPoint {
public:
	ControlPoint(){ pos.setZero(); active = true; anchorForce = 0; }
	ControlPoint( Eigen::Vector3d pos_ ) : pos(pos_) { active = true; }

	Eigen::Vector3d pos;
	bool active;
	MovingAnchor* anchorForce;
};


class MovingAnchor : public Force {
public:

	MovingAnchor( int idx_, std::shared_ptr<ControlPoint> p_ ) : idx(idx_), point(p_) { point -> anchorForce = this; }

	void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep ){ weight = 1000.f; }
	void computeDi( int dof );
	void update( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;

	int idx;
	std::shared_ptr<ControlPoint> point;

protected:

}; // end class force


} // end namespace admm

#endif




