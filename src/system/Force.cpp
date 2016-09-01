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
	rest_length = disp.norm();
	weight = sqrt(stiffness);
}

void Spring::computeDi( int dof ){

	SparseMatrix<double> newDi;
	newDi.resize(3,dof);
	const int col0 = 3 * idx0; const int col1 = 3 * idx1;
	newDi.coeffRef(0,col0) = 1.0; newDi.coeffRef(1,col0+1) = 1.0; newDi.coeffRef(2,col0+2) = 1.0;
	newDi.coeffRef(0,col1) = -1.0; newDi.coeffRef(1,col1+1) = -1.0; newDi.coeffRef(2,col1+2) = -1.0;
	setDi( newDi );	
}

void Spring::update( double dt, const VectorXd &Dx, VectorXd &u, VectorXd &z ) const {

	// Computing Di * x + ui
	int Di_rows = getDi()->rows();
	VectorXd Dix = Dx.segment( global_idx, Di_rows );
	VectorXd ui = u.segment( global_idx, Di_rows );
	VectorXd DixPlusUi = Dix + ui;

	// Analytical update using projection p
	double DixPlusUi_norm = DixPlusUi.norm();
	VectorXd DixPlusUi_normed = DixPlusUi / DixPlusUi_norm;

	VectorXd p = rest_length * DixPlusUi_normed;

	VectorXd zi = ( 1.0 / (weight*weight + stiffness) ) * (stiffness*p + weight*weight*(DixPlusUi));
	ui += ( Dix - zi );
	u.segment( global_idx, Di_rows ) = ui;
	z.segment( global_idx, Di_rows ) = zi;
	
}

/*
double Spring::getEnergy( const Eigen::VectorXd& Dx, Eigen::VectorXd& grad ) const {
	int Di_rows = getDi()->rows();
	VectorXd Dix = Dx.segment( global_idx, Di_rows );
	double energy = 0.5 * stiffness * pow( rest_length - Dix.norm() , 2.0 );
	grad.segment( global_idx, Di_rows ) = stiffness * ( rest_length - Dix.norm() ) * Dix.normalized();
	return energy;
}
*/




