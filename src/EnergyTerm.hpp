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

#ifndef ADMM_FORCE_H
#define ADMM_FORCE_H 1

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <memory>
#include <iostream>

namespace admm {


//
//	Lame constants
//
class Lame {
public:
	static Lame rubber(){ return Lame(10000000,0.499); } // true rubber
	static Lame soft_rubber(){ return Lame(10000000,0.399); } // fun rubber!

	double mu, lambda;
	double bulk_modulus() const { return lambda + (2.0/3.0)*mu; }

	// k: Youngs (Pa), measure of stretch
	// v: Poisson, measure of incompressibility
	Lame( double k, double v ) :
		mu(k/(2.0*(1.0+v))),
		lambda(k*v/((1.0+v)*(1.0-2.0*v))) {}

	// Use custom mu, lambda
	Lame(){}
};


//
//	Base class tets: Linear (non-corotated) elastic
//
class EnergyTerm {
private:
	typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;
	typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SparseMat;
	int g_index; // global idx (starting row of reduction matrix)

public:

	virtual ~EnergyTerm() {}

	// Called by the solver to create the global reduction and weight matrices
	inline void get_reduction( std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights );

	// Called by the solver for a local step update
	inline void update( const SparseMat &D, const VecX &x, VecX &z, VecX &u );

	// Computes energy of the force (used for debugging)
	inline double energy( const SparseMat &D, const VecX &x );

	// Compute energy and gradient for a first-order opt. solver
	inline double gradient( const SparseMat &D, const VecX &x, VecX &grad );

	// Dimension of deformation gradient.
	virtual int get_dim() const = 0;

	// Return the scalar weight of the energy term
	virtual double get_weight() const = 0;

protected:

	// Get a local reduction matrix
	virtual void get_reduction( std::vector< Eigen::Triplet<double> > &triplets ) = 0;

	// Proximal update
	virtual void prox( VecX &zi ) = 0;

	// Returns energy of the force
	virtual double energy( const VecX &F ) = 0;

	// Computes a first-order update (energy and gradient)
	virtual double gradient( const VecX &F, VecX &grad ) = 0;

}; // end class EnergyTerm

//
//  Implementation
//

inline void EnergyTerm::get_reduction( std::vector< Eigen::Triplet<double> > &triplets, std::vector<double> &weights ){
	std::vector< Eigen::Triplet<double> > temp_triplets;
	get_reduction( temp_triplets );
	int n_trips = temp_triplets.size();
	g_index = weights.size();
	for( int i=0; i<n_trips; ++i ){
		const Eigen::Triplet<double> &trip = temp_triplets[i];
		triplets.emplace_back( trip.row()+g_index, trip.col(), trip.value() );
	}
	int dim = get_dim();
	double w = get_weight();
	if( w <= 0.0 ){
		throw std::runtime_error("**EnergyTerm::get_reduction Error: Some weight leq 0");
	}
	for( int i=0; i<dim; ++i ){ weights.emplace_back( w ); }
}

inline void EnergyTerm::update( const SparseMat &D, const VecX &x, VecX &z, VecX &u ){
	int dof = x.rows();
	int dim = get_dim();
	VecX Dix = D.block(g_index,0,dim,dof)*x;
	VecX ui = u.segment(g_index,dim);
	VecX zi = Dix + ui;
	prox( zi );
	ui += (Dix - zi);
	u.segment(g_index,dim) = ui;
	z.segment(g_index,dim) = zi;
}

inline double EnergyTerm::energy( const SparseMat &D, const VecX &x ){
	int dof = x.rows();
	int dim = get_dim();
	VecX Dix = D.block(g_index,0,dim,dof)*x;
	return energy( Dix );
}

inline double EnergyTerm::gradient( const SparseMat &D, const VecX &x, VecX &grad ){
	int dof = x.rows();
	int dim = get_dim();
	VecX Dix = D.block(g_index,0,dim,dof)*x;
	VecX gradi = grad.segment(g_index,dim);
	double e = gradient( Dix, gradi );
	grad.segment(g_index,dim) = gradi;
	return e;
}


} // end namespace admm

#endif




