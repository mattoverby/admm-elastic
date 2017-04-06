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

#include "TetForce.hpp"

using namespace admm;
using namespace Eigen;

namespace admm {
namespace helper {

	static inline void init_tet_force( int *idx, const Eigen::VectorXd &x, double &volume, Eigen::Matrix<double,4,3> &B, Eigen::Matrix<double,3,3> &edges_inv ){
		using namespace Eigen;

		// Edge matrix
		Matrix<double,3,3> edges;
		Vector3d v0( x[ idx[0]*3 ], x[ idx[0]*3+1 ], x[ idx[0]*3+2 ] );
		Vector3d v1( x[ idx[1]*3 ], x[ idx[1]*3+1 ], x[ idx[1]*3+2 ] );
		Vector3d v2( x[ idx[2]*3 ], x[ idx[2]*3+1 ], x[ idx[2]*3+2 ] );
		Vector3d v3( x[ idx[3]*3 ], x[ idx[3]*3+1 ], x[ idx[3]*3+2 ] );
		edges.col(0) = v1 - v0;
		edges.col(1) = v2 - v0;
		edges.col(2) = v3 - v0;
		edges_inv = edges.inverse();

		// Xg is beginning state		
		Matrix<double,3,3> Xg = edges;

		// D matrix is I dunno
		Matrix<double,4,3> D;
		D(0,0) = -1; D(0,1) = -1; D(0,2) = -1;
		D(1,0) =  1; D(1,1) =  0; D(1,2) =  0;
		D(2,0) =  0; D(2,1) =  1; D(2,2) =  0;
		D(3,0) =  0; D(3,1) =  0; D(3,2) =  1;

		// B is used to create Ai
		B = D * Xg.inverse();

		// Volume is used for weight
		volume = fabs( (v0-v3).dot( (v1-v3).cross(v2-v3) ) ) / 6.0;
	}

	static inline void init_tet_Di( int *idx, const Eigen::Matrix<double,4,3> &B, const int dof, std::vector<Eigen::Triplet<double> > &triplets ){
		using namespace Eigen;
		int constraint_idx = triplets.size();
		Matrix<double,3,4> Bt = B.transpose();
		const int col0 = 3 * idx[0];
		const int col1 = 3 * idx[1];
		const int col2 = 3 * idx[2];
		const int col3 = 3 * idx[3];
		const int rows[3] = { 0, 3, 6 };
		const int cols[4] = { col0, col1, col2, col3 };
		for( int r=0; r<3; ++r ){
			for( int c=0; c<4; ++c ){
				double value = Bt(r,c);
				for( int j=0; j<3; ++j ){
					triplets.push_back( Eigen::Triplet<double>(rows[r]+j+constraint_idx,cols[c]+j,value) );
				}
			}
		}
	}

	// Projection, Singular Values, SVD's U, SVD's V transpose
	static inline void oriented_svd( const Eigen::Matrix3d &F, Eigen::Vector3d &S, Eigen::Matrix3d &U, Eigen::Matrix3d &Vt ){
		using namespace Eigen;

		JacobiSVD< Matrix3d > svd( F, ComputeFullU | ComputeFullV );
		S = svd.singularValues();
		U = svd.matrixU();
		Vt = svd.matrixV().transpose();

		Matrix3d J = Matrix3d::Identity(); J(2,2)=-1.0;

		// Check for inversion: U
		if( U.determinant() < 0.0 ){
			U = U * J;
			S[2] *= -1.0;
		}

		// Check for inversion: V
		if( Vt.determinant() < 0.0 ){
			Vt = J * Vt;
			S[2] *= -1.0;
		}

	} // end oriented svd

}
}

//
//	LinearTetStrain
//


void LinearTetStrain::initialize( const VectorXd &x, const VectorXd &v, const VectorXd &masses, const double timestep ){
	helper::init_tet_force( idx, x, volume, B, edges_inv );

	// Calculate weight
	weight = sqrtf(stiffness)*sqrtf(volume);// * weight_scale;
}

void LinearTetStrain::get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights ){
	global_idx = weights.size();
	helper::init_tet_Di( idx, B, x.size(), triplets );
	for( int i=global_idx; i<triplets.size(); ++i ){
		weights.push_back( weight );
	}
}

void LinearTetStrain::project( double dt, const VectorXd &Dx, VectorXd &u, VectorXd &z ) const {
	typedef Matrix<double,9,1> Vector9d;
	Vector9d Dix = Dx.segment<9>( global_idx );
	Vector9d ui = u.segment<9>( global_idx );
	Vector9d DixPlusUi = Dix+ui;

	// Computing F (rearranging terms from 9x1 vector DixPlusUi to make a 3x3)
	Matrix<double,3,3> F = Map<Matrix<double,3,3> >(DixPlusUi.data());

	// Get some singular values
	JacobiSVD< Matrix<double,3,3> > svd(F, ComputeFullU | ComputeFullV);
	Vector3d S = svd.singularValues();
	S[0]=1.0;S[1]=1.0;S[2]=1.0;
	if( F.determinant() < 0.0 ){ S[2] = -1.0; }

	// Reconstruct with new singular values
	Matrix<double,3,3> proj = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();
	Vector9d p = Map<Vector9d>(proj.data());

	// Update zi and ui
	double k = stiffness*volume;
	Vector9d zi = ( k*p + weight*weight*(DixPlusUi) ) / (weight*weight + k);

	ui.noalias() += ( Dix - zi );
	u.segment<9>( global_idx ) = ui;
	z.segment<9>( global_idx ) = zi;
}

//
// LinearTetVolume
//


void TetVolume::initialize( const VectorXd &x, const VectorXd &v, const VectorXd &masses, const double timestep ){
	helper::init_tet_force( idx, x, rest_volume, B, edges_inv );
	weight = sqrtf(stiffness)*sqrtf(rest_volume);// * weight_scale;
}

void TetVolume::get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights ){
	global_idx = weights.size();
	helper::init_tet_Di( idx, B, x.size(), triplets );
	for( int i=global_idx; i<triplets.size(); ++i ){
		weights.push_back( weight );
	}
}

void TetVolume::project( double dt, const VectorXd &Dx, VectorXd &u, VectorXd &z ) const {

	typedef Matrix<double,9,1> Vector9d;
	Vector9d Dix = Dx.segment<9>( global_idx );
	Vector9d ui = u.segment<9>( global_idx );
	Vector9d DixPlusUi = Dix+ui;

	// Computing F (rearranging terms from 9x1 vector DixPlusUi to make a 3x3)
	Matrix<double,3,3> F = Map<Matrix<double,3,3> >(DixPlusUi.data());

	// Get some singular values
	JacobiSVD< Matrix<double,3,3> > svd(F, ComputeFullU | ComputeFullV);
	Vector3d S = svd.singularValues();
	Eigen::Vector3d d(0,0,0);

	for(int i = 0; i < 4; i++){
		double detS = S[0] * S[1] * S[2];
		double f = detS - std::min( std::max(detS,limit_min) , limit_max );
		Eigen::Vector3d g( S[1]*S[2] , S[0]*S[2] , S[0]*S[1] );
		d = -((f - g.dot(d)) / g.dot(g)) * g;
		S = svd.singularValues() + d;
	}

	if( F.determinant() < 0.0 ){ S[2] = -1.0; }

	// Reconstruct with new singular values
	Matrix<double,3,3> proj = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();
	Vector9d p = Map<Vector9d>(proj.data());

	// Update zi and ui
	double k = stiffness*rest_volume;
	Vector9d zi = ( k*p + weight*weight*(DixPlusUi) ) / (weight*weight + k);

	ui.noalias() += ( Dix - zi );
	u.segment<9>( global_idx ) = ui;
	z.segment<9>( global_idx ) = zi;

}

//
//	NeoHookeanTet
//

double NHProx::energyDensity( const cppoptlib::Vector<double> &Sigma ) const {
	double Sig_det = (Sigma[0]*Sigma[1]*Sigma[2]);
	double I_1 = Sigma[0]*Sigma[0]+Sigma[1]*Sigma[1]+Sigma[2]*Sigma[2];
	double I_3 = Sig_det*Sig_det;
	double log_I3 = log( I_3 );
	double t1 = 0.5 * mu * ( I_1 - log_I3 - 3.0 );
	double t2 = 0.125 * lambda * log_I3 * log_I3;
	double r = t1 + t2;
	return r;
}

// Compute objective function (prox operator)
double NHProx::value(const cppoptlib::Vector<double> &x) {
	if( x[0]<0.0 || x[1]<0.0 || x[2]<0.0 ){ return std::numeric_limits<float>::max(); }
	double r = energyDensity( x );
	double r2 = (k*0.5) * (x-Sigma_init).squaredNorm();
	return ( scaleConst*r + r2 );
}

void NHProx::gradient(const cppoptlib::Vector<double> &x, cppoptlib::Vector<double> &grad){
	double detSigma = x[0]*x[1]*x[2];
	if( detSigma <= 0.0 ){
		grad = VectorXd::Ones(3) * std::numeric_limits<float>::max();
	} else {
		Eigen::Vector3d invSigma(1.0/x[0],1.0/x[1],1.0/x[2]);
		grad = scaleConst*(mu * (x - invSigma) + lambda * log(detSigma) * invSigma) + k*(x-Sigma_init);
	}  
}

void NHProx::hessian(const cppoptlib::Vector<double> &x, cppoptlib::Matrix<double> &hessian){
	Eigen::Vector3d Sigma(x[0],x[1],x[2]);
	Eigen::Vector3d invSigma(1.0/x[0],1.0/x[1],1.0/x[2]);
	double detSigma = Sigma[0] * Sigma[1] * Sigma[2];
	Eigen::Matrix3d invSigmaSqMat;
	invSigmaSqMat.setZero();
	invSigmaSqMat(0,0) = 1.0 / (Sigma[0]*Sigma[0]);
	invSigmaSqMat(1,1) = 1.0 / (Sigma[1]*Sigma[1]);
	invSigmaSqMat(2,2) = 1.0 / (Sigma[2]*Sigma[2]);
	Eigen::Matrix3d I;
	I.setIdentity();
	Eigen::Matrix3d hess = mu*(I - 2.0*invSigmaSqMat) + lambda*log(detSigma)*invSigmaSqMat
	                       + lambda * invSigma * invSigma.transpose() + k*I;
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			hessian(i,j) = hess(i,j);
		}
	}
}

//
//	St. VK tet
//

double StVKProx::energyDensity( Vector3d &Sigma ) const {
	Eigen::Vector3d I(1.0,1.0,1.0);
	Eigen::Vector3d Sigma2( Sigma[0]*Sigma[0], Sigma[1]*Sigma[1], Sigma[2]*Sigma[2] );

	// Strain tensor
	Eigen::Vector3d st = ( 0.5 * ( Sigma2 - I ) );
	double st_tr2 = v3trace(st)*v3trace(st);
	double r = ( mu * ddot( st, st ) + ( lambda * 0.5 * st_tr2 ) );
	return r;
}

// Compute objective function (prox operator)
double StVKProx::value(const cppoptlib::Vector<double> &x) {
	if( x[0]<0.0 || x[1]<0.0 || x[2]<0.0 ){ return std::numeric_limits<float>::max(); }
	Eigen::Vector3d Sigma(x[0],x[1],x[2]);
	double r = energyDensity( Sigma );
	double r2 = (k*0.5) * (Sigma-this->Sigma_init).squaredNorm();
	return (r+r2);
}

void StVKProx::gradient(const cppoptlib::Vector<double> &x, cppoptlib::Vector<double> &grad){
	Eigen::Vector3d term1;
	term1[0] = mu * x[0]*(x[0]*x[0] - 1.0);
	term1[1] = mu * x[1]*(x[1]*x[1] - 1.0);
	term1[2] = mu * x[2]*(x[2]*x[2] - 1.0);
	Eigen::Vector3d term2;
	term2 = 0.5 * lambda * ( x.dot(x) - 3.0 ) * x;
	grad = term1 + term2 + k*(x-Sigma_init);
}

//
//	The Hyper Elastic Force class
//

void HyperElasticTet::initialize( const VectorXd &x, const VectorXd &v, const VectorXd &masses, const double timestep ){
	helper::init_tet_force( idx, x, volume, B, edges_inv );

	double stiff = std::min(mu,lambda); // what should k be?
	weight = sqrtf(stiff)*sqrtf(volume);
	stvkprox = std::shared_ptr<StVKProx>( new StVKProx(mu,lambda,stiff) );
	nhprox = std::shared_ptr<NHProx>( new NHProx(mu,lambda,stiff) );
}

void HyperElasticTet::get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights ){
	global_idx = weights.size();
	helper::init_tet_Di( idx, B, x.size(), triplets );
	for( int i=global_idx; i<triplets.size(); ++i ){
		weights.push_back( weight );
	}
}

void HyperElasticTet::project( double dt, const VectorXd &Dx, VectorXd &u, VectorXd &z ) const {

	typedef Matrix<double,9,1> Vector9d;
	Vector9d Dix = Dx.segment<9>( global_idx );
	Vector9d ui = u.segment<9>( global_idx );
	Vector9d DixPlusUi = Dix+ui;

	// Computing F (rearranging terms from 9x1 vector DixPlusUi to make a 3x3)
	Matrix3d F = Map<Matrix3d>(DixPlusUi.data());

	// SVD of deform grad (model reduction, faster solve)
	Vector3d S0; Matrix3d U, Vt;
	helper::oriented_svd( F, S0, U, Vt );

	// Initialize the problem
	nhprox->setSigma0( S0 );
	stvkprox->setSigma0( S0 );

	// Initial guess
	Eigen::VectorXd x2 = last_prox_result;

	// Initial guess needs positive entries
	if( x2[2] < 0.0 ){ x2[2] *= -1.0; }

	// If everything is very low, this is our collapsed-node test case
	else if( fabs( x2[0] ) < 1.e-3 && fabs( x2[1] ) < 1.e-3 && fabs( x2[2] ) < 1.e-3 ){
		x2[0] = 1.e-3; x2[1] = 1.e-3; x2[2] = 1.e-3;
	}

	// Local minimize with L-BFGS
	switch( type ){
		case 0: solver->minimize(*(nhprox.get()), x2); break;
		case 1: solver->minimize(*(stvkprox.get()), x2); break;
	}

	// Reconstruct with new singular values
	last_prox_result = x2;
	Matrix3d proj = U * x2.asDiagonal() * Vt;
	Vector9d zi = Map<Vector9d>(proj.data());

	// Update global vars
	ui.noalias() += ( Dix - zi );
	u.segment<9>( global_idx ) = ui;
	z.segment<9>( global_idx ) = zi;
}


