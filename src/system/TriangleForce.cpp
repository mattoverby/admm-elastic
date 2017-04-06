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

	Eigen::Matrix<double,3,2> basis;
	Eigen::Matrix<double,3,2> edges;

	basis.col(0) = n1; basis.col(1) = n2;
	edges.col(0) = e12; edges.col(1) = e13;
	
	Matrix<double,2,2> Xg = (basis.transpose() * edges);
	Matrix<double,3,3> X123;
	X123.col(0) = x1; X123.col(1) = x2; X123.col(2) = x3;
	
	B = D * Xg.inverse();

	area = std::abs((basis.transpose() * edges).determinant() / 2.0f);
	weight = sqrtf(stiffness) * sqrtf(area);
}


void LimitedTriangleStrain::get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights ){
	global_idx = weights.size();
	int cols[3] = { 3*id0, 3*id1, 3*id2 };
	for( int i=0; i<3; ++i ){
		for( int j=0; j<3; ++j ){
			triplets.push_back( Triplet<double>(i+global_idx, cols[j]+i, B(j,0) ) );
			triplets.push_back( Triplet<double>(3+i+global_idx, cols[j]+i, B(j,1) ) );
		}
	}
	for( int i=0; i<6; ++i ){ weights.push_back( weight ); }
}


void LimitedTriangleStrain::project( double dt, const VectorXd &Dx, VectorXd &u, VectorXd &z ) const {

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

	if( strain_limiting ){
		double l_col0 = zi.head<3>().norm();
		double l_col1 = zi.tail<3>().norm();
		if( l_col0 < limit_min ){ zi.head<3>() *= ( limit_min / fmaxf( l_col0, 1e-6 ) ); }
		if( l_col1 < limit_min ){ zi.tail<3>() *= ( limit_min / fmaxf( l_col1, 1e-6 ) ); }
		if( l_col0 > limit_max ){ zi.head<3>() *= ( limit_max / fmaxf( l_col0, 1e-6 ) ); }
		if( l_col1 > limit_max ){ zi.tail<3>() *= ( limit_max / fmaxf( l_col1, 1e-6 ) ); }
	}

	// update u and z
	ui.noalias() += ( Dix - zi );
	u.segment<6>( global_idx ) = ui;
	z.segment<6>( global_idx ) = zi;
}


//
//	Fung Skin Model
//

inline double FungProx::energyDensity( Vector3d &Sigma ) const {
	double I_1 = Sigma[0]*Sigma[0]+Sigma[1]*Sigma[1]+Sigma[2]*Sigma[2];
	double t1 = mu/(b*2.0);
	double t2 = exp( b*(I_1-3.0) ) - 1.0;
	if( !std::isfinite(t2) ){ return std::numeric_limits<float>::max(); }
	double r = (t1*t2);
	return r;
}

//inline double FungProx::incompress( Vector2d &Sigma ) const {
//	return -mu*Sigma[0]*Sigma[1]+mu;
//}

// Compute objective function (prox operator)
double FungProx::value(const cppoptlib::Vector<double> &x) {
	if( x[0]<=0.0 || x[1]<=0.0 ){ return std::numeric_limits<float>::max(); }
	if( std::isnan( x[0] ) || std::isnan( x[1] ) ){ printf("\nBAD X: %f %f\n", x[0], x[1] ); }
	Eigen::Vector2d Sigma(x[0],x[1]);
	Eigen::Vector3d Sigma3(x[0],x[1],1.0/(x[0]*x[1]));
	double r0 = energyDensity( Sigma3 );
//	double r1 = incompress( Sigma );
	double r2 = (k*0.5) * (Sigma-this->Sigma_init).squaredNorm();
//printf("\nmu: %f", mu);
	return (r0+r2);
}

void FungProx::gradient(const cppoptlib::Vector<double> &x, cppoptlib::Vector<double> &grad){
	//finiteGradient(x,grad); return;
	double detSigma = x[0]*x[1];
	const double minval = std::numeric_limits<float>::min();
	if( std::abs(x[0])<minval || std::abs(x[1])<minval ){
		grad = VectorXd::Ones(2) * std::numeric_limits<float>::max();
		return;
	}

	double sig3 = 1.0/(x[0]*x[1]);
	double I_1 = (x[0]*x[0]+x[1]*x[1]+sig3*sig3);
	double t1 = 0.5 * mu * exp( b*( I_1-3.0 ) );
	Eigen::Vector2d t2 = k*(x-this->Sigma_init);
	grad[0] = t1*(2.0*x[0]-2.0/(x[0]*x[0]*x[0]*x[1]*x[1]) ) + t2[0];
	grad[1] = t1*(2.0*x[1]-2.0/(x[1]*x[1]*x[1]*x[0]*x[0]) ) + t2[1];

}

void FungTriangle::initialize( const VectorXd &x, const VectorXd &v, const VectorXd &masses, const double timestep ){

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

	Eigen::Matrix<double,3,2> basis;
	Eigen::Matrix<double,3,2> edges;
	basis.col(0) = n1; basis.col(1) = n2;
	edges.col(0) = e12; edges.col(1) = e13;
	
	Matrix<double,2,2> Xg = (basis.transpose() * edges);
	Matrix<double,3,3> X123;
	X123.col(0) = x1; X123.col(1) = x2; X123.col(2) = x3;
	
	B = D * Xg.inverse();

//	weight = sqrt( std::min(a,c) );

	area = std::abs((basis.transpose() * edges).determinant() / 2.0f);
	weight = sqrt(mu) * sqrt(area);
	double k = mu;//(weight*weight)/area;
//	weight = sqrt(mu);
//	double k = mu/area;
	fungprox = std::unique_ptr<FungProx>( new FungProx(mu,k) );

}


void FungTriangle::get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights ){
	global_idx = weights.size();
	int cols[3] = { 3*id0, 3*id1, 3*id2 };
	for( int i=0; i<3; ++i ){
		for( int j=0; j<3; ++j ){
			triplets.push_back( Triplet<double>(i+global_idx, cols[j]+i, B(j,0) ) );
			triplets.push_back( Triplet<double>(3+i+global_idx, cols[j]+i, B(j,1) ) );
		}
	}
	for( int i=0; i<6; ++i ){ weights.push_back( weight ); }
}



void FungTriangle::project( double dt, const VectorXd &Dx, VectorXd &u, VectorXd &z ) const {

	typedef Matrix<double,6,1> Vector6d;
	Vector6d Dix = Dx.segment<6>( global_idx );
	Vector6d ui = u.segment<6>( global_idx );
	Vector6d DixPlusUi = Dix+ui;

	// Computing F (rearranging terms from 6x1 vector AixPlusUi to make a 3x2)
	Matrix<double,3,2> F = Map<Matrix<double,3,2> >(DixPlusUi.data());
	JacobiSVD<Matrix<double,3,2> > svd(F, ComputeFullU | ComputeFullV);
	cppoptlib::Vector<double> x2 = svd.singularValues();

	// Minimize
	fungprox->setSigma0( Eigen::Vector2d(x2[0],x2[1]) );
	solver->minimize(*(fungprox.get()), x2);

//	printf("\n%f %f", x2(0), x2(1) );
	// Incompressibility
//	if( x2(0) < 1 ){ x2(0) = 1; }
//	if( x2(1) < 1 ){ x2(1) = 1; }

	// Reform F
	Matrix<double,3,2> Diag; Diag.setZero();
	Diag.block<2,2>(0,0) = x2.asDiagonal();
	F = svd.matrixU() * Diag * svd.matrixV().transpose();
	Vector6d zi = Map<Vector6d>(F.data());

	// update u and z
	ui.noalias() += ( Dix - zi );
	u.segment<6>( global_idx ) = ui;
	z.segment<6>( global_idx ) = zi;
}


static inline double aclamp( double v, double min, double max ){
	v = ( v < max ? v : max );
	v = ( v > min ? v : min );
	return v;
}

void TriArea::project( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const {

	typedef Matrix<double,6,1> Vector6d;
	Vector6d Dix = Dx.segment<6>( global_idx );
	Vector6d ui = u.segment<6>( global_idx );
	Vector6d DixPlusUi = Dix+ui;

	// Computing F (rearranging terms from 6x1 vector AixPlusUi to make a 3x2)
	Matrix<double,3,2> F = Map<Matrix<double,3,2> >(DixPlusUi.data());
	JacobiSVD<Matrix<double,3,2> > svd(F, ComputeFullU | ComputeFullV);

	// Compute the singular value decomposition
	Eigen::Vector2d S = svd.singularValues();
	Eigen::Vector2d d(0.0f, 0.0f);
	for (int i = 0; i < iters; ++i) {
		double v = S(0) * S(1);
		double f = v - aclamp(v, limit_min, limit_max);
		Eigen::Vector2d g(S(1), S(0));
		d = -((f - g.dot(d)) / g.dot(g)) * g;
		S = svd.singularValues() + d;
	}

	// Reconstruct F and compute projection
	Matrix<double,3,2> Diag;
	Diag.setZero();
	Diag.block<2,2>(0,0) = S.asDiagonal();
	F = svd.matrixU() * Diag * svd.matrixV().transpose();
	Vector6d p = Map<Vector6d>(F.data());
	
	// Update zi and ui
	double k = stiffness*area;
	Vector6d zi = ( k*p + weight*weight*(DixPlusUi) ) / ( weight*weight + k );

	// update u and z
	ui.noalias() += ( Dix - zi );
	u.segment<6>( global_idx ) = ui;
	z.segment<6>( global_idx ) = zi;

}


