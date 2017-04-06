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

#include "BendForce.hpp"

using namespace Eigen;
using namespace admm;


void BendForce::initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const VectorXd &masses, const double timestep ){
		
	Eigen::Vector3d x0( x[ idx[0]*3 ], x[ idx[0]*3+1 ], x[ idx[0]*3+2 ] );
	Eigen::Vector3d x1( x[ idx[1]*3 ], x[ idx[1]*3+1 ], x[ idx[1]*3+2 ] );
	Eigen::Vector3d x2( x[ idx[2]*3 ], x[ idx[2]*3+1 ], x[ idx[2]*3+2 ] );
	Eigen::Vector3d x3( x[ idx[3]*3 ], x[ idx[3]*3+1 ], x[ idx[3]*3+2 ] );

	weight = sqrt(stiffness);

	Eigen::Vector3d xA = x0 - x2;
	Eigen::Vector3d xB = x1 - x2;
	Eigen::Vector3d xC = Eigen::Vector3d(0,0,0);
	Eigen::Vector3d xD = x3 - x2;
	
	double area1 = 0.5*(xA.cross(xD)).norm();
	double area2 = 0.5*(xD.cross(xB)).norm();
	
	double hA = 2.0 * area1 / xD.norm();
	double hB = 2.0 * area2 / xD.norm();
	
	
	Eigen::Vector3d nA = (xA - xC).cross(xA - xD);
	Eigen::Vector3d nB = (xB - xD).cross(xB - xC);
	Eigen::Vector3d nC = (xC - xB).cross(xC - xA);
	Eigen::Vector3d nD = (xD - xA).cross(xD - xB);
	
	alpha[0] = hB / (hA + hB);
	alpha[1] = hA / (hA + hB);
	alpha[2]= -nD.norm() / ( nC.norm() + nD.norm() );
	alpha[3] = -nC.norm() / ( nC.norm() + nD.norm() );
	
	double lambda = (2.0/3.0) * ( hA + hB ) / pow( hA * hB , 2.0) * (xD - xC).norm() * stiffness;
	
	Eigen::Matrix3d I;
	I.setIdentity();
	
	Eigen::MatrixXd jac(9,9);
	
	jac.block(0,0,3,3) = -lambda * alpha[0] * alpha[0] * I;
	jac.block(0,3,3,3) = -lambda * alpha[0] * alpha[3] * I;
	jac.block(0,6,3,3) = -lambda * alpha[0] * alpha[1] * I;
	jac.block(3,0,3,3) = -lambda * alpha[3] * alpha[0] * I;
	jac.block(3,3,3,3) = -lambda * alpha[3] * alpha[3] * I;
	jac.block(3,6,3,3) = -lambda * alpha[3] * alpha[1] * I;
	jac.block(6,0,3,3) = -lambda * alpha[1] * alpha[0] * I;
	jac.block(6,3,3,3) = -lambda * alpha[1] * alpha[3] * I;
	jac.block(6,6,3,3) = -lambda * alpha[1] * alpha[1] * I;
}

void BendForce::get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights ){

	global_idx = weights.size();
	for( int i=0; i<9; ++i ){ weights.push_back(weight); }

	int col0 = 3*idx[0];
	int col1 = 3*idx[1];
	int col2 = 3*idx[2];
	int col3 = 3*idx[3];
	
	// Setting nonzero terms of first row of Di
//	newDi.coeffRef(0,col0) = 1.0;
//	newDi.coeffRef(1,col0+1) = 1.0;
//	newDi.coeffRef(2,col0+2) = 1.0;
	triplets.push_back( Triplet<double>( 0+global_idx, col0, 1.0 ) );
	triplets.push_back( Triplet<double>( 1+global_idx, col0+1, 1.0 ) );
	triplets.push_back( Triplet<double>( 2+global_idx, col0+2, 1.0 ) );
	
//	newDi.coeffRef(0,col2) = -1.0;
//	newDi.coeffRef(1,col2+1) = -1.0;
//	newDi.coeffRef(2,col2+2) = -1.0;
	triplets.push_back( Triplet<double>( 0+global_idx, col2, -1.0 ) );
	triplets.push_back( Triplet<double>( 1+global_idx, col2+1, -1.0 ) );
	triplets.push_back( Triplet<double>( 2+global_idx, col2+2, -1.0 ) );

	// Setting nonzero terms of second row of Di
//	newDi.coeffRef(3,col3) = 1.0;
//	newDi.coeffRef(4,col3+1) = 1.0;
//	newDi.coeffRef(5,col3+2) = 1.0;
	triplets.push_back( Triplet<double>( 3+global_idx, col3, 1.0 ) );
	triplets.push_back( Triplet<double>( 4+global_idx, col3+1, 1.0 ) );
	triplets.push_back( Triplet<double>( 5+global_idx, col3+2, 1.0 ) );

//	newDi.coeffRef(3,col2) = -1.0;
//	newDi.coeffRef(4,col2+1) = -1.0;
//	newDi.coeffRef(5,col2+2) = -1.0;
	triplets.push_back( Triplet<double>( 3+global_idx, col2, -1.0 ) );
	triplets.push_back( Triplet<double>( 4+global_idx, col2+1, -1.0 ) );
	triplets.push_back( Triplet<double>( 5+global_idx, col2+2, -1.0 ) );
	
	// Setting nonzero terms of third row of Di
//	newDi.coeffRef(6,col1) = 1.0;
//	newDi.coeffRef(7,col1+1) = 1.0;
//	newDi.coeffRef(8,col1+2) = 1.0;
	triplets.push_back( Triplet<double>( 6+global_idx, col1, 1.0 ) );
	triplets.push_back( Triplet<double>( 7+global_idx, col1+1, 1.0 ) );
	triplets.push_back( Triplet<double>( 8+global_idx, col1+2, 1.0 ) );
	
//	newDi.coeffRef(6,col2) = -1.0;
//	newDi.coeffRef(7,col2+1) = -1.0;
//	newDi.coeffRef(8,col2+2) = -1.0;
	triplets.push_back( Triplet<double>( 6+global_idx, col2, -1.0 ) );
	triplets.push_back( Triplet<double>( 7+global_idx, col2+1, -1.0 ) );
	triplets.push_back( Triplet<double>( 8+global_idx, col2+2, -1.0 ) );
	

}


inline void BendForce::computeUsingProjection( Vector9d& p, Vector9d& Dix) const{

	Eigen::Vector3d c1 = Dix.segment<3>(0);
	Eigen::Vector3d c2 = Dix.segment<3>(3);
	Eigen::Vector3d c3 = Dix.segment<3>(6);
	Eigen::Vector3d lam = 2.0 * ( alpha[0]*c1 + alpha[3]*c2 + alpha[1]*c3 ) / ( alpha[0]*alpha[0] + alpha[3]*alpha[3] + alpha[1]*alpha[1] );
	p.segment<3>(0) = c1 - 0.5*alpha[0]*lam;
	p.segment<3>(3) = c2 - 0.5*alpha[3]*lam;
	p.segment<3>(6) = c3 - 0.5*alpha[1]*lam;
	
}

void BendForce::project( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const{

	// Computing Di * x + ui
	Vector9d Dix = Dx.segment<9>( global_idx );
	Vector9d ui = u.segment<9>( global_idx );
	Vector9d DixPlusUi = Dix+ui;

	Vector9d p;
	computeUsingProjection(p,DixPlusUi);
	Vector9d zi = ( 1.0 / (weight*weight + stiffness) ) * (stiffness*p + weight*weight*(DixPlusUi));		
	
	ui.noalias() += ( Dix - zi );
	u.segment<9>( global_idx ) = ui;
	z.segment<9>( global_idx ) = zi;

}


