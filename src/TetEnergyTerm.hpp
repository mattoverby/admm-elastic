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

#ifndef ADMM_TETENERGYTERM_H
#define ADMM_TETENERGYTERM_H

#include "XuSpline.hpp"
#include "EnergyTerm.hpp"
#include "MCL/Newton.hpp"
#include "MCL/LBFGS.hpp"
#include "MCL/NonLinearCG.hpp"

namespace admm {


//
//	Creates a bunch of tet energy terms from a tet mesh
//
template <typename IN_SCALAR, typename TYPE>
inline void create_tets_from_mesh( std::vector< std::shared_ptr<EnergyTerm> > &energyterms,
	const IN_SCALAR *verts, const int *inds, int n_tets, const Lame &lame, const int vertex_offset ){
	typedef Eigen::Matrix<int,4,1> Vec4i;
	typedef Eigen::Matrix<double,3,1> Vec3;
	for( int i=0; i<n_tets; ++i ){
		Vec4i tet(inds[i*4+0], inds[i*4+1], inds[i*4+2], inds[i*4+3]);
		std::vector<Vec3> tetverts = {
			Vec3( verts[tet[0]*3+0], verts[tet[0]*3+1], verts[tet[0]*3+2] ),
			Vec3( verts[tet[1]*3+0], verts[tet[1]*3+1], verts[tet[1]*3+2] ),
			Vec3( verts[tet[2]*3+0], verts[tet[2]*3+1], verts[tet[2]*3+2] ),
			Vec3( verts[tet[3]*3+0], verts[tet[3]*3+1], verts[tet[3]*3+2] )
		};
		tet += Vec4i(1,1,1,1)*vertex_offset;
		energyterms.emplace_back( std::make_shared<TYPE>(tet, tetverts, lame) );
	}
} // end create from mesh


//
//	The tet energy term base class
//
class TetEnergyTerm : public EnergyTerm {
protected:
	typedef Eigen::Matrix<int,4,1> Vec4i;
	typedef Eigen::Matrix<double,3,1> Vec3;
	typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;
	Vec4i tet;
	Lame lame;
	double volume;
	double weight;
	Eigen::Matrix3d edges_inv;

public:
	int get_dim() const { return 9; }
	double get_weight() const { return weight; }

	TetEnergyTerm( const Vec4i &tet_, const std::vector<Vec3> &verts, const Lame &lame_ );

	void get_reduction( std::vector< Eigen::Triplet<double> > &triplets );

	// Unless derived from uses linear tet strain (no volume conservation)
	virtual void prox( VecX &zi );
	virtual double energy( const VecX &F );
	virtual double gradient( const VecX &F, VecX &grad );

}; // end class TetEnergyTerm

//
//	Neo-Hookean Tet
//
class NeoHookeanTet : public TetEnergyTerm {
public:
	typedef Eigen::Matrix<double,3,3> Mat3;
	class NHProx : public mcl::optlib::Problem<double,3> {
	public:
		double mu, lambda, k;
		Vec3 x0;
		double energy_density(const Vec3 &x) const;
		double value(const Vec3 &x);
		double gradient(const Vec3 &x, Vec3 &grad);
		bool converged(const Vec3 &x0, const Vec3 &x1, const Vec3 &grad){
			return ( grad.norm() < 1e-6 || (x0-x1).norm() < 1e-6 );
		}
	} problem;
	mcl::optlib::LBFGS<double,3> solver;

	NeoHookeanTet( const Vec4i &tet_, const std::vector<Vec3> &verts, const Lame &lame_ );
	void prox( VecX &zi );
	double energy( const VecX &F );
	double gradient( const VecX &F, VecX &grad );
};


//
//	St-VK Tet
//
class StVKTet : public TetEnergyTerm {
public:
	typedef Eigen::Matrix<double,3,3> Mat3;
	class StVKProx : public mcl::optlib::Problem<double,3> {
	public:
		double mu, lambda, k;
		Vec3 x0;
		double energy_density(const Vec3 &x) const;
		double value(const Vec3 &x);
		double gradient(const Vec3 &x, Vec3 &grad);
		static inline double ddot( Vec3 &a, Vec3 &b ) { return (a*b.transpose()).trace(); }
		static inline double v3trace( const Vec3 &v ) { return v[0]+v[1]+v[2]; }
		bool converged(const Vec3 &x0, const Vec3 &x1, const Vec3 &grad){
			return ( grad.norm() < 1e-6 || (x0-x1).norm() < 1e-6 );
		}
	} problem;
	mcl::optlib::LBFGS<double,3> solver;

	StVKTet( const Vec4i &tet_, const std::vector<Vec3> &verts, const Lame &lame_ );
	void prox( VecX &zi );
	double energy( const VecX &F );
	double gradient( const VecX &F, VecX &grad );
};



//
//	Spline Tet:
// 	Nonlinear Material Design Using Principal Stretches (2015)
//	Hongyi Xu, Funshing Sin, Yufeng Zhu, Jernej Barbic
//
class SplineTet : public TetEnergyTerm {
public:
	// The actual proximal objective function we are minimizing in the
	// local step (subject to a quadratic constraint for ADMM coupling constraint)
	typedef Eigen::Matrix<double,3,3> Mat3;
	class SplineProx : public mcl::optlib::Problem<double,3> {
	public:
		Vec3 x0;
		double k;
		std::shared_ptr<xu::Spline> spline;
		double energy_density(const Vec3 &x) const;
		double value(const Vec3 &x);
		double gradient(const Vec3 &x, Vec3 &grad);
		bool converged(const Vec3 &x0, const Vec3 &x1, const Vec3 &grad){
			return ( grad.norm() < 1e-6 || (x0-x1).norm() < 1e-6 );
		}
	} problem;
	mcl::optlib::LBFGS<double,3> solver;

	// Defaults to NeoHookean if this constructor is used.
	SplineTet( const Vec4i &tet_, const std::vector<Vec3> &verts, const Lame &lame_ );
	SplineTet( const Vec4i &tet_, const std::vector<Vec3> &verts, const Lame &lame_,
		std::shared_ptr<xu::Spline> spline );
	void prox( VecX &zi );
	double energy( const VecX &F );
	double gradient( const VecX &F, VecX &grad );
};


} // end namespace admm

#endif




