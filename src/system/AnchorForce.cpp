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

#include "AnchorForce.hpp"

using namespace admm;
using namespace Eigen;


//
//	Anchor Constraint
//


void StaticAnchor::initialize( const VectorXd &x, const VectorXd &v, const VectorXd &masses, const double timestep ){
	pos[0] = x[ idx*3 + 0 ];
	pos[1] = x[ idx*3 + 1 ];
	pos[2] = x[ idx*3 + 2 ];
}

void StaticAnchor::get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights ){
	global_idx = weights.size();
	const int col = 3*idx;
	for( int i=0; i<3; ++i ){
		weights.push_back( weight );
		triplets.push_back( Eigen::Triplet<double>( global_idx+i, col+i, 1.0 ) );
	}
}

void StaticAnchor::project( double dt, const VectorXd &Dx, VectorXd &u, VectorXd &z ) const {

	Vector3d Dix = Dx.segment<3>( global_idx );
	Vector3d ui = u.segment<3>( global_idx );

	// project zi and ui
	ui += ( Dix - pos );
	u.segment<3>( global_idx ) = ui;
	z.segment<3>( global_idx ) = pos;
}

//
//	Moving Anchor
//

void MovingAnchor::get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights ){
	global_idx = weights.size();
	const int col = 3*idx;
	for( int i=0; i<3; ++i ){
		weights.push_back( weight );
		triplets.push_back( Eigen::Triplet<double>( global_idx+i, col+i, 1.0 ) );
	}
}


void MovingAnchor::project( double dt, const VectorXd &Dx, VectorXd &u, VectorXd &z ) const {
	int Di_rows = 3;
	VectorXd Dix = Dx.segment( global_idx, Di_rows );
	VectorXd ui = u.segment( global_idx, Di_rows );
	Eigen::Vector3d zi(Di_rows);

	// If active, project on to the constraint manifold
	if( point -> active ){
		zi = point -> pos;
	} else {
		zi = Dix + ui;
		point -> pos = Dx.segment( global_idx, 3 );
	}

	// project zi and ui
	ui += ( Dix - zi );
	u.segment( global_idx, Di_rows ) = ui;
	z.segment( global_idx, Di_rows ) = zi;
}


