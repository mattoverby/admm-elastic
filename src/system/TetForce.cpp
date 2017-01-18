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

#include "TetForce.hpp"

using namespace admm;
using namespace Eigen;

//
//	LinearTetStrain
//


void LinearTetStrain::initialize( const VectorXd &x, const VectorXd &v, const VectorXd &masses, const double timestep ){
	helper::init_tet_force( idx, x, volume, B, edges_inv );

	// Calculate weight
	weight = sqrtf(stiffness)*sqrtf(volume) * weight_scale;
}

void LinearTetStrain::computeDi( int dof ){
	SparseMatrix<double> newDi;
	helper::init_tet_Di( idx, B, dof, newDi );
	setDi( newDi );
}

void LinearTetStrain::update( double dt, const VectorXd &Dx, VectorXd &u, VectorXd &z ) const {

	// Computing Di * x + ui
	int Di_rows = getDi()->rows();
	VectorXd Dix = Dx.segment( global_idx, Di_rows );
	VectorXd ui = u.segment( global_idx, Di_rows );
	VectorXd DixPlusUi = Dix + ui;

	// Computing F (rearranging terms from 9x1 vector DixPlusUi to make a 3x3)
	Matrix<double,3,3> F;
	F(0,0) = DixPlusUi(0); F(1,0) = DixPlusUi(1); F(2,0) = DixPlusUi(2);
	F(0,1) = DixPlusUi(3); F(1,1) = DixPlusUi(4); F(2,1) = DixPlusUi(5);
	F(0,2) = DixPlusUi(6); F(1,2) = DixPlusUi(7); F(2,2) = DixPlusUi(8);

	// Get some singular values
	JacobiSVD< Matrix<double,3,3> > svd(F, ComputeFullU | ComputeFullV);
	Vector3d S = svd.singularValues();

	S[0]=1.0;S[1]=1.0;S[2]=1.0;

	if( F.determinant() < 0.0 ){ S[2] = -1.0; }

	// Reconstruct with new singular values
	Matrix<double,3,3> proj = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();

	// Proj needs to be 9d vector
	VectorXd p; p.resize(9);
	p(0) = proj(0,0); p(3) = proj(0,1); p(6) = proj(0,2);
	p(1) = proj(1,0); p(4) = proj(1,1); p(7) = proj(1,2);
	p(2) = proj(2,0); p(5) = proj(2,1); p(8) = proj(2,2);

	// Update zi and ui
	double k = stiffness*volume;
	VectorXd zi = ( k*p + weight*weight*(DixPlusUi) ) / (weight*weight + k);

	ui += ( Dix - zi );

	u.segment( global_idx, Di_rows ) = ui;
	z.segment( global_idx, Di_rows ) = zi;

}

//
// LinearTetVolume
//

void LinearTetVolume::initialize( const VectorXd &x, const VectorXd &v, const VectorXd &masses, const double timestep ){
	helper::init_tet_force( idx, x, volume, B, edges_inv );
	weight = sqrtf(stiffness)*sqrtf(volume);
}

void LinearTetVolume::computeDi( int dof ){
	SparseMatrix<double> newDi;
	helper::init_tet_Di( idx, B, dof, newDi );
	setDi( newDi );
}

void LinearTetVolume::update( double dt, const VectorXd &Dx, VectorXd &u, VectorXd &z ) const {

	// Computing Di * x + ui
	int Di_rows = getDi()->rows();
	VectorXd Dix = Dx.segment( global_idx, Di_rows );
	VectorXd ui = u.segment( global_idx, Di_rows );
	VectorXd DixPlusUi = Dix + ui;

	// Computing F (rearranging terms from 9x1 vector DixPlusUi to make a 3x3)
	Matrix<double,3,3> F;
	F(0,0) = DixPlusUi(0); F(1,0) = DixPlusUi(1); F(2,0) = DixPlusUi(2);
	F(0,1) = DixPlusUi(3); F(1,1) = DixPlusUi(4); F(2,1) = DixPlusUi(5);
	F(0,2) = DixPlusUi(6); F(1,2) = DixPlusUi(7); F(2,2) = DixPlusUi(8);

	// Get some singular values
	JacobiSVD< Matrix<double,3,3> > svd(F, ComputeFullU | ComputeFullV);
	Vector3d S = svd.singularValues();
	Eigen::Vector3d d(0,0,0);
	
	for(int i = 0; i < 4; i++){

		double detS = S[0] * S[1] * S[2];
		double f = detS - std::min( std::max(detS,rangeMin) , rangeMax );
		Eigen::Vector3d g( S[1]*S[2] , S[0]*S[2] , S[0]*S[1] );
		d = -((f - g.dot(d)) / g.dot(g)) * g;
		S = svd.singularValues() + d;
	}
	
	if( svd.matrixU().determinant()*svd.matrixV().determinant() < 0.0f ){
		S[2] = -S[2];
	}
	
	// Reconstruct with new singular values
	Matrix<double,3,3> proj = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();

	// Proj needs to be 9d vector
	VectorXd p; p.resize(9);
	p(0) = proj(0,0); p(3) = proj(0,1); p(6) = proj(0,2);
	p(1) = proj(1,0); p(4) = proj(1,1); p(7) = proj(1,2);
	p(2) = proj(2,0); p(5) = proj(2,1); p(8) = proj(2,2);

	// Update zi and ui
	double k = stiffness*volume;
	VectorXd zi = ( k*p + weight*weight*(DixPlusUi) ) / (weight*weight + k);

	ui += ( Dix - zi );

	u.segment( global_idx, Di_rows ) = ui;
	z.segment( global_idx, Di_rows ) = zi; 
	
}


//
// Anisotropic tet
//

void AnisotropicTet::initialize( const VectorXd &x, const VectorXd &v, const VectorXd &masses, const double timestep ){
	helper::init_tet_force( idx, x, volume, B, edges_inv );
	weight = sqrtf(stiffness)*sqrtf(volume);
}

void AnisotropicTet::computeDi( int dof ){
	SparseMatrix<double> newDi;
	helper::init_tet_Di( idx, B, dof, newDi );
	setDi( newDi );
}

void AnisotropicTet::update( double dt, const VectorXd &Dx, VectorXd &u, VectorXd &z ) const {

	// Computing Di * x + ui
	int Di_rows = getDi()->rows();
	VectorXd Dix = Dx.segment( global_idx, Di_rows );
	VectorXd ui = u.segment( global_idx, Di_rows );
	VectorXd DixPlusUi = Dix + ui;

	// Computing F (rearranging terms from 9x1 vector DixPlusUi to make a 3x3)
	Matrix<double,3,3> F;
	F(0,0) = DixPlusUi(0); F(1,0) = DixPlusUi(1); F(2,0) = DixPlusUi(2);
	F(0,1) = DixPlusUi(3); F(1,1) = DixPlusUi(4); F(2,1) = DixPlusUi(5);
	F(0,2) = DixPlusUi(6); F(1,2) = DixPlusUi(7); F(2,2) = DixPlusUi(8);
	
	double FeNorm = (F*e).norm();
	double alpha = (1.0 - FeNorm ) / ( (weight*weight)/(k*volume) + FeNorm );
	//double alpha = ( (F*e).norm() - (F*e).squaredNorm() ) / ( (F*e).squaredNorm() + (weight*weight/(2.0*k*volume))*(F*e*e.transpose()).norm());
	
	F += alpha*F*eeT;
	
	Eigen::VectorXd zi(9);
	zi(0) = F(0,0);  zi(3) = F(0,1);  zi(6) = F(0,2);
	zi(1) = F(1,0);  zi(4) = F(1,1);  zi(7) = F(1,2);
	zi(2) = F(2,0);  zi(5) = F(2,1);  zi(8) = F(2,2);
	
	ui += (Dix - zi);
	
	u.segment( global_idx, Di_rows ) = ui;
	z.segment( global_idx, Di_rows ) = zi;
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
	// Return max float (large number) that won't cause nans
	if( x[0]<0.0 || x[1]<0.0 || x[2]<0.0 ){ return std::numeric_limits<float>::max(); }
	double r = energyDensity( x );
	double r2 = (k*0.5) * (x-Sigma_init).squaredNorm();
	return ( scaleConst*r + r2 );
}

void NHProx::gradient(const cppoptlib::Vector<double> &x, cppoptlib::Vector<double> &grad){
	double detSigma = x[0]*x[1]*x[2];
	if( detSigma <= 0.0 ){
//		grad.setZero();
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
	// Return max float (large number) that won't cause nans
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

	// Max or min?
//	double stiff = std::max(mu,lambda);
	double stiff = std::min(mu,lambda);
	weight = sqrtf(stiff)*sqrtf(volume);
	double k = (weight*weight/volume);
	stvkprox = std::shared_ptr<StVKProx>( new StVKProx(mu,lambda,k,volume) );
	nhprox = std::shared_ptr<NHProx>( new NHProx(mu,lambda,k,volume) );
}

void HyperElasticTet::computeDi( int dof ){
	SparseMatrix<double> newDi;
	helper::init_tet_Di( idx, B, dof, newDi );
	setDi( newDi );
}

void HyperElasticTet::update( double dt, const VectorXd &Dx, VectorXd &u, VectorXd &z ) const {

	// Computing Di * x + ui
	int Di_rows = getDi()->rows();
	VectorXd Dix = Dx.segment( global_idx, Di_rows );
	VectorXd ui = u.segment( global_idx, Di_rows );
	VectorXd DixPlusUi = Dix + ui;

	// Computing F (rearranging terms from 9x1 vector DixPlusUi to make a 3x3)
	Matrix<double,3,3> F;
	F(0,0) = DixPlusUi(0); F(1,0) = DixPlusUi(1); F(2,0) = DixPlusUi(2);
	F(0,1) = DixPlusUi(3); F(1,1) = DixPlusUi(4); F(2,1) = DixPlusUi(5);
	F(0,2) = DixPlusUi(6); F(1,2) = DixPlusUi(7); F(2,2) = DixPlusUi(8);

	// SVD of deform grad (model reduction, faster solve)
	Vector3d S0; Matrix3d U, Vt;
	helper::oriented_svd( F, S0, U, Vt );

	// Initialize the problem
	nhprox->setSigma0( S0 );
	stvkprox->setSigma0( S0 );

	// Initial guess
	cppoptlib::Vector<double> x2 = last_prox_result;

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

	// Remake the deform grad
	last_prox_result = x2;
	Matrix3d Zi = U * x2.asDiagonal() * Vt;
	VectorXd zi(9);
	zi(0) = Zi(0,0); zi(1) = Zi(1,0); zi(2) = Zi(2,0);
	zi(3) = Zi(0,1); zi(4) = Zi(1,1); zi(5) = Zi(2,1); 
	zi(6) = Zi(0,2); zi(7) = Zi(1,2); zi(8) = Zi(2,2);

	#ifdef PDADMM_VERIFY
	if( Zi.determinant() < 0.0 || !( x2[0]>0.0 && x2[1]>0.0 && x2[2]>0.0 ) ){
		std::cout << "\n\nNH Tet inversion: det F:" << Zi.determinant() << "\nnew F:\n" << Zi << "\nnew sigma: \n" << x2 << "\n\n" << std::endl;
		std::cout << "init sigma: " << S0[0] << " " << S0[1] << " " << S0[2] << std::endl;

		Eigen::Vector3d v0(0,0,0);
		Eigen::Vector3d v1(zi(0),zi(1),zi(2));
		Eigen::Vector3d v2(zi(3),zi(4),zi(5));
		Eigen::Vector3d v3(zi(6),zi(7),zi(8));
		double volumeCurrent = fabs( (v0-v3).dot( (v1-v3).cross(v2-v3) ) ) / 6.0;

		std::cout << "original volume was .. " << volume << std::endl;
		std::cout << "current volume is .. " << volumeCurrent << std::endl;
		exit(0);
	}
	#endif

	// Update global vars
	ui += ( Dix - zi );
	u.segment( global_idx, Di_rows ) = ui;
	z.segment( global_idx, Di_rows ) = zi;
}


