// Copyright (c) 2017, University of Minnesota
// 
// lbfgssolver Uses the BSD 2-Clause License (http://www.opensource.org/licenses/BSD-2-Clause)
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

#ifndef ADMM_TRIANGLEFORCE_H
#define ADMM_TRIANGLEFORCE_H 1

#include "Force.hpp"

namespace admm {

//
//	LimitedTriangleStrain
//
class LimitedTriangleStrain : public Force {
public:
	LimitedTriangleStrain( int id0_, int id1_, int id2_, double stiffness_, double limit_min_, double limit_max_, bool strain_limiting_=true ) :
		id0(id0_), id1(id1_), id2(id2_), stiffness(stiffness_), limit_min(limit_min_), limit_max(limit_max_), strain_limiting(strain_limiting_) {}
	virtual void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep );
	virtual void get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights );
	virtual void project( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;

	int id0, id1, id2;
	double stiffness, limit_min, limit_max;
	Eigen::Matrix<double,3,2> B;
	double area;
	bool strain_limiting;

}; // end class limited triangle strain

// Proximal Operator for Fung
class FungProx : public cppoptlib::Problem<double> {
public:
	FungProx(double mu_, double k_ ) : mu(mu_), b(1.0), k(k_) {}
	void setSigma0( const Eigen::Vector2d &Sigma_init_ ){ Sigma_init=Sigma_init_; }

	inline double energyDensity( Eigen::Vector3d &Sigma ) const;
	double value(const cppoptlib::Vector<double> &x);
	void gradient(const cppoptlib::Vector<double> &x, cppoptlib::Vector<double> &grad);
	Eigen::Vector2d Sigma_init;
	double mu, b, k;

}; // end class StVKProx

//
//	FungTriangle for Skin
//
class FungTriangle : public Force {
public:
	FungTriangle( int id0_, int id1_, int id2_, double mu_, double limit_min_, double limit_max_ ) :
		id0(id0_), id1(id1_), id2(id2_), mu(mu_), limit_min(limit_min_), limit_max(limit_max_) {
		solver = std::unique_ptr< cppoptlib::ISolver<double, 1> >( new cppoptlib::lbfgssolver<double> );
		solver->settings_.maxIter = 10;
		solver->settings_.gradTol = 1e-6;
	}
	void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep );
	void get_selector( const Eigen::VectorXd &x, std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights );
	void project( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;

	std::unique_ptr< cppoptlib::ISolver<double, 1> > solver;
	std::unique_ptr<FungProx> fungprox;
	int id0, id1, id2;
	double mu, limit_min, limit_max;
	double area;
	Eigen::Matrix<double,3,2> B;

}; // end class limited triangle strain

class TriArea : public LimitedTriangleStrain {
public:
	TriArea( int id0_, int id1_, int id2_, double stiffness_, int iters_, double limit_min_, double limit_max_ ) :
		LimitedTriangleStrain( id0_, id1_, id2_, stiffness_, limit_min_, limit_max_ ), iters(iters_) {}
	void project( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;
	int iters;
};

} // end namespace admm

#endif




