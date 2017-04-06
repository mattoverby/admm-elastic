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

#ifndef ADMM_TETFORCE_H
#define ADMM_TETFORCE_H 1

#include "Force.hpp"

namespace admm {

//
//	Linear Tet
//

class LinearTetStrain : public Force {
public:
	LinearTetStrain( int idx0_, int idx1_, int idx2_, int idx3_, double stiffness_, double weight_scale_=1.f ) :
		stiffness(stiffness_), volume(0.0), weight_scale(weight_scale_)
		{ idx[0]=idx0_; idx[1]=idx1_; idx[2]=idx2_; idx[3]=idx3_; }

	void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep );
	void project( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;
	void get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights );


	int idx[4];
	double stiffness, volume, weight_scale;
	Eigen::Matrix3d edges_inv; // used for piola stress
	Eigen::Matrix<double,4,3> B;

}; // end class LinearTetStrain

//
//	Linear Tet Volume
//
class TetVolume : public Force {
public:
	TetVolume( int idx0_, int idx1_, int idx2_, int idx3_, double stiffness_, double limit_min_, double limit_max_ ) :
		stiffness(stiffness_), rest_volume(0.0), limit_min(limit_min_), limit_max(limit_max_) { idx[0]=idx0_; idx[1]=idx1_; idx[2]=idx2_; idx[3]=idx3_; }

	void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep );
	void project( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;
	void get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights );

	int idx[4];
	Eigen::Matrix3d edges_inv; // used for piola stress
	double stiffness, rest_volume;
	double limit_min, limit_max;
	Eigen::Matrix<double,4,3> B;

}; // end class TetVolume

//
//	NeoHookean Hyper Elastic
//

// Proximal Operator for neohookean
class NHProx : public cppoptlib::Problem<double> {
public:
	NHProx(double scaleConst_, double mu_, double lambda_, double k_ ) : scaleConst(scaleConst_), mu(mu_), lambda(lambda_), k(k_) {}
	NHProx(double mu_, double lambda_, double k_) : mu(mu_), lambda(lambda_), k(k_) { scaleConst = 1.0; }
	void setSigma0( Eigen::Vector3d &Sigma_init_ ){ Sigma_init=Sigma_init_; }
	double energyDensity( const cppoptlib::Vector<double> &sigma) const;
	double value(const cppoptlib::Vector<double> &x);
	void gradient(const cppoptlib::Vector<double> &x, cppoptlib::Vector<double> &grad);
	void hessian(const cppoptlib::Vector<double> &x, cppoptlib::Matrix<double> &hess);
	Eigen::Vector3d Sigma_init;
	double scaleConst, mu, lambda, k;

}; // end class nhprox

//
//	Saint Venant-Kirchhoff Hyper Elastic
//

// Proximal Operator for StVK
class StVKProx : public cppoptlib::Problem<double> {
public:
	StVKProx(double mu_, double lambda_, double k_) : mu(mu_), lambda(lambda_), k(k_) {}
	void setSigma0( Eigen::Vector3d &Sigma_init_ ){ Sigma_init=Sigma_init_; }
	double energyDensity( Eigen::Vector3d &Sigma ) const;
	double value(const cppoptlib::Vector<double> &x);
	void gradient(const cppoptlib::Vector<double> &x, cppoptlib::Vector<double> &grad);
	Eigen::Vector3d Sigma_init;
	double mu, lambda, k;
	static inline double ddot( Eigen::Vector3d &a, Eigen::Vector3d &b ) { return (a*b.transpose()).trace(); }
	static inline double v3trace( const Eigen::Vector3d &v ) { return v[0]+v[1]+v[2]; }
}; // end class StVKProx

//
//	Hyper Elastic Tet Forces
//
//	Type is either:
//		nh	(0)
//		stvk	(1)
//
//	It's probably better to be able to pass in a proximal operator as an argument instead of the elastic
//	type for extensibility. However, this is easier to do for now.
//
class HyperElasticTet : public Force {
public:
	HyperElasticTet( int idx0_, int idx1_, int idx2_, int idx3_, double mu_, double lambda_, int max_iterations, std::string type_ ) :
		mu(mu_), lambda(lambda_) {
		idx[0]=idx0_; idx[1]=idx1_; idx[2]=idx2_; idx[3]=idx3_;
		type = 0; if( type_=="stvk" || type_=="1" ){ type=1; }

		// Set up the local solver
		solver = std::unique_ptr< cppoptlib::ISolver<double, 1> >( new cppoptlib::lbfgssolver<double> );
		solver->settings_.maxIter = max_iterations;
		solver->settings_.gradTol = 1e-8;
		last_prox_result.resize(3);
		last_prox_result.fill(1.0);
	}

	void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep );
	void project( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;
	void get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights );

	std::shared_ptr<NHProx> nhprox;
	std::shared_ptr<StVKProx> stvkprox;
	std::unique_ptr< cppoptlib::ISolver<double, 1> > solver;


	int idx[4];
	int type;
	double mu, lambda, volume;
	Eigen::Matrix3d edges_inv; // used for piola stress
	Eigen::Matrix<double,4,3> B;
	mutable cppoptlib::Vector<double> last_prox_result;

}; // end class HyperElastic


} // end namespace admm

#endif




