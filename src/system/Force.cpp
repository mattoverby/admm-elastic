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

#include "Force.hpp"

using namespace admm;
using namespace Eigen;

//
//	Spring Force
//

void Spring::initialize( const VectorXd &x, const VectorXd &v, const VectorXd &masses, const double timestep ){

	Vector3d x_0( x[idx0*3], x[idx0*3+1], x[idx0*3+2] );
	Vector3d x_1( x[idx1*3], x[idx1*3+1], x[idx1*3+2] );
	Vector3d disp = x_0 - x_1;
	double length = disp.norm();
//	if( length <= 0.0 ){ length = 1e-16; }
	rest_length = length;
	weight = sqrt(stiffness);
}

void Spring::get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights ){
	global_idx = weights.size();
	const int col0 = 3 * idx0; const int col1 = 3 * idx1;

	for( int i=0; i<3; ++i ){
		triplets.push_back( Triplet<double>( i+global_idx, col0+i, 1.0 ) );
		triplets.push_back( Triplet<double>( i+global_idx, col1+i, -1.0 ) );
	}

	for( int i=0; i<3; ++i ){ weights.push_back(weight); }
}

void Spring::project( double dt, const VectorXd &Dx, VectorXd &u, VectorXd &z ) const {

	// Computing Di * x + ui
	Vector3d Dix = Dx.segment<3>( global_idx );
	Vector3d ui = u.segment<3>( global_idx );
	Vector3d DixUi = Dix + ui;

	// Analytical update using projection p
	double DixUi_norm = DixUi.norm();
	Vector3d DixUi_normed = DixUi / DixUi_norm;
	if( DixUi_norm <= 0.0 ){ DixUi_normed.setZero(); }

	Vector3d p = rest_length * DixUi_normed;
	Vector3d zi = ( 1.0 / (weight*weight + stiffness) ) * (stiffness*p + weight*weight*(DixUi));

	ui += ( Dix - zi );
	u.segment<3>( global_idx ) = ui;
	z.segment<3>( global_idx ) = zi;

}

