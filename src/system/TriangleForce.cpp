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

#include "TriangleForce.hpp"

using namespace admm;
using namespace Eigen;

//
//	TriangleStrain
//


void LimitedTriangleStrain::initialize( const VectorXd &x, const VectorXd &v, const VectorXd &masses, const double timestep ){

	assert(3*id0+2 < x.size());
	assert(3*id1+2 < x.size());
	assert(3*id2+2 < x.size());

	Vector3d x1( x(3*id0+0), x(3*id0+1), x(3*id0+2) );
	Vector3d x2( x(3*id1+0), x(3*id1+1), x(3*id1+2) );
	Vector3d x3( x(3*id2+0), x(3*id2+1), x(3*id2+2) );

	Matrix<double,3,2> D;
	D(0,0) = -1; D(0,1) = -1;
	D(1,0) =  1; D(1,1) =  0;
	D(2,0) =  0; D(2,1) =  1;

	Vector3d e12 = x2 - x1;
	Vector3d e13 = x3 - x1;
	Vector3d n1 = e12.normalized();
	Vector3d n2 = (e13 - e13.dot(n1)*n1).normalized();

	Matrix<double,3,2> basis;
	Matrix<double,3,2> edges;
	basis.col(0) = n1; basis.col(1) = n2;
	edges.col(0) = e12; edges.col(1) = e13;
	
	Matrix<double,2,2> Xg = (basis.transpose() * edges);
	Matrix<double,3,3> X123;
	X123.col(0) = x1; X123.col(1) = x2; X123.col(2) = x3;
	
	B = D * Xg.inverse();

	area = std::abs((basis.transpose() * edges).determinant() / 2.0f);
	weight = sqrt(stiffness)*sqrt(area);

}


void LimitedTriangleStrain::computeDi( int dof ){

	SparseMatrix<double> newDi;
	newDi.resize(6,dof);
	newDi.setZero();
	
	int col0 = 3*id0;
	int col1 = 3*id1;
	int col2 = 3*id2;
	
	newDi.coeffRef(0,col0) = B(0,0);
	newDi.coeffRef(1,col0+1) = B(0,0);
	newDi.coeffRef(2,col0+2) = B(0,0);
	
	newDi.coeffRef(3,col0) = B(0,1);
	newDi.coeffRef(4,col0+1) = B(0,1);
	newDi.coeffRef(5,col0+2) = B(0,1);
	
	newDi.coeffRef(0,col1) = B(1,0);
	newDi.coeffRef(1,col1+1) = B(1,0);
	newDi.coeffRef(2,col1+2) = B(1,0);
	
	newDi.coeffRef(3,col1) = B(1,1);
	newDi.coeffRef(4,col1+1) = B(1,1);
	newDi.coeffRef(5,col1+2) = B(1,1);
	
	newDi.coeffRef(0,col2) = B(2,0);
	newDi.coeffRef(1,col2+1) = B(2,0);
	newDi.coeffRef(2,col2+2) = B(2,0);
	
	newDi.coeffRef(3,col2) = B(2,1);
	newDi.coeffRef(4,col2+1) = B(2,1);
	newDi.coeffRef(5,col2+2) = B(2,1);

	setDi( newDi );
}


void LimitedTriangleStrain::update( double dt, const VectorXd &Dx, VectorXd &u, VectorXd &z ) const {

	typedef Matrix<double,6,1> Vector6d;
	Vector6d Dix = Dx.segment<6>( global_idx );
	Vector6d ui = u.segment<6>( global_idx );
	Vector6d DixPlusUi = Dix+ui;

	// Computing F (rearranging terms from 6x1 vector AixPlusUi to make a 3x2)
	Matrix<double,3,2> F = Map<Matrix<double,3,2> >(DixPlusUi.data());
		
	// Compute the singular value decomposition
	JacobiSVD<Matrix<double,3,2> > svd(F, ComputeFullU | ComputeFullV);
	
	// Constructing the matrix T
	Matrix<double,3,2> T = svd.matrixU().leftCols<2>() * svd.matrixV().transpose();
		
	Vector6d p = Map<Vector6d>(T.data());
	
	// Update zi and ui
	double k = stiffness*area;
	Vector6d zi = ( k*p + weight*weight*(DixPlusUi) ) / ( weight*weight + k );

	//std::cout << "Zi is .. \n" << Zi << std::endl;
	
	// Divide out the weight
	//Zi /= weight;

//4. For cloth it's actually more realistic to only limit the
//axis-aligned strain but not the shear strain. That's easy to do:
//instead of taking the SVD of F and limiting its singular values, just
//scale the columns of F so their lengths lie in the specified range.

	double l_col0 = zi.head<3>().norm();
	double l_col1 = zi.tail<3>().norm();
	if( l_col0 < limit_min ){ zi.head<3>() *= ( limit_min / fmaxf( l_col0, 1e-6 ) ); }
	if( l_col1 < limit_min ){ zi.tail<3>() *= ( limit_min / fmaxf( l_col1, 1e-6 ) ); }
	if( l_col0 > limit_max ){ zi.head<3>() *= ( limit_max / fmaxf( l_col0, 1e-6 ) ); }
	if( l_col1 > limit_max ){ zi.tail<3>() *= ( limit_max / fmaxf( l_col1, 1e-6 ) ); }
	
	// update u and z
	ui.noalias() += ( Dix - zi );
	u.segment<6>( global_idx ) = ui;
	z.segment<6>( global_idx ) = zi;
}


//
//	PDTriangleStrain
//


void PDTriangleStrain::initialize( const VectorXd &x, const VectorXd &v, const VectorXd &masses, const double timestep ){

	assert(3*id0+2 < x.size());
	assert(3*id1+2 < x.size());
	assert(3*id2+2 < x.size());

	Vector3d x1( x(3*id0+0), x(3*id0+1), x(3*id0+2) );
	Vector3d x2( x(3*id1+0), x(3*id1+1), x(3*id1+2) );
	Vector3d x3( x(3*id2+0), x(3*id2+1), x(3*id2+2) );

	Matrix<double,3,2> D;
	D(0,0) = -1; D(0,1) = -1;
	D(1,0) =  1; D(1,1) =  0;
	D(2,0) =  0; D(2,1) =  1;

	Vector3d e12 = x2 - x1;
	Vector3d e13 = x3 - x1;
	Vector3d n1 = e12.normalized();
	Vector3d n2 = (e13 - e13.dot(n1)*n1).normalized();

	Matrix<double,3,2> basis;
	Matrix<double,3,2> edges;
	basis.col(0) = n1; basis.col(1) = n2;
	edges.col(0) = e12; edges.col(1) = e13;
	
	Matrix<double,2,2> Xg = (basis.transpose() * edges);
	Matrix<double,3,3> X123;
	X123.col(0) = x1; X123.col(1) = x2; X123.col(2) = x3;
	
	B = D * Xg.inverse();

	area = std::abs((basis.transpose() * edges).determinant() / 2.0f);
	weight = sqrt(stiffness)*sqrt(area);

}


void PDTriangleStrain::computeDi( int dof ){

	SparseMatrix<double> newDi;
	newDi.resize(6,dof);
	newDi.setZero();
	
	int col0 = 3*id0;
	int col1 = 3*id1;
	int col2 = 3*id2;
	
	newDi.coeffRef(0,col0) = B(0,0);
	newDi.coeffRef(1,col0+1) = B(0,0);
	newDi.coeffRef(2,col0+2) = B(0,0);
	
	newDi.coeffRef(3,col0) = B(0,1);
	newDi.coeffRef(4,col0+1) = B(0,1);
	newDi.coeffRef(5,col0+2) = B(0,1);
	
	newDi.coeffRef(0,col1) = B(1,0);
	newDi.coeffRef(1,col1+1) = B(1,0);
	newDi.coeffRef(2,col1+2) = B(1,0);
	
	newDi.coeffRef(3,col1) = B(1,1);
	newDi.coeffRef(4,col1+1) = B(1,1);
	newDi.coeffRef(5,col1+2) = B(1,1);
	
	newDi.coeffRef(0,col2) = B(2,0);
	newDi.coeffRef(1,col2+1) = B(2,0);
	newDi.coeffRef(2,col2+2) = B(2,0);
	
	newDi.coeffRef(3,col2) = B(2,1);
	newDi.coeffRef(4,col2+1) = B(2,1);
	newDi.coeffRef(5,col2+2) = B(2,1);

	setDi( newDi );
}


void PDTriangleStrain::update( double dt, const VectorXd &Dx, VectorXd &u, VectorXd &z ) const {

	int Di_rows = getDi()->rows();
	VectorXd Dix = Dx.segment( global_idx, Di_rows );
	VectorXd ui = u.segment( global_idx, Di_rows );
	VectorXd DixPlusUi = Dix+ui;

	// Computing F (rearranging terms from 6x1 vector AixPlusUi to make a 3x2)
	Matrix<double,3,2> F;
	F(0,0) = DixPlusUi(0);
	F(1,0) = DixPlusUi(1);
	F(2,0) = DixPlusUi(2);
	F(0,1) = DixPlusUi(3);
	F(1,1) = DixPlusUi(4);
	F(2,1) = DixPlusUi(5);
		
	// Compute the singular value decomposition
	JacobiSVD<Matrix<double,3,2> > svd(F, ComputeFullU | ComputeFullV);
	
	// Setting the singular values to 1
	Vector2d S = svd.singularValues();

	// Do strain limiting now
	if( S(0) < limit_min ){
		S(0) = limit_min;
	} else if( S(0) > limit_max ){
		S(0) = limit_max;
	}
	
	if( S(1) < limit_min ){
		S(1) = limit_min;
	} else if( S(1) > limit_max ){
		S(1) = limit_max;
	}
	
	// Creating a 3x2 matrix with the (0,0) and (1,1) entries being the tweaked singular values
	Matrix<double,3,2> Diag;
	Diag.setZero();
	Diag.block<2,2>(0,0) = S.asDiagonal();
	
	// Constructing the matrix T
	Matrix<double,3,2> T;
	T = svd.matrixU() * Diag * svd.matrixV().transpose();
		
	VectorXd p(6);
	p(0) = T(0,0);
	p(1) = T(1,0);
	p(2) = T(2,0);
	p(3) = T(0,1);
	p(4) = T(1,1);
	p(5) = T(2,1);
	
	// Update zi and ui
	double k = stiffness*area;
	VectorXd zi = ( k*p + weight*weight*(DixPlusUi) ) / ( weight*weight + k );
	
	// update u and z
	ui += ( Dix - zi );
	u.segment( global_idx, Di_rows ) = ui;
	z.segment( global_idx, Di_rows ) = zi;
}

