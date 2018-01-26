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

#include "TetEnergyTerm.hpp"
#include <iostream>
#include "FastSVD.hpp"

using namespace admm;
using namespace Eigen;

//
//	TetEnergyTerm
//

TetEnergyTerm::TetEnergyTerm( const Vec4i &tet_, const std::vector<Vec3> &verts, const Lame &lame_ ) :
	tet(tet_), lame(lame_), volume(0.0), weight(0.0) {

	// Compute inv rest pose
	Matrix<double,3,3> edges; // B
	edges.col(0) = verts[1] - verts[0];
	edges.col(1) = verts[2] - verts[0];
	edges.col(2) = verts[3] - verts[0];
	edges_inv = edges.inverse();

	volume = (edges).determinant() / 6.0f;
	if( volume < 0 ){
		throw std::runtime_error("**TetEnergyTerm Error: Inverted initial tet");
	}

	double k = lame.bulk_modulus(); // projective dynamics weight
	weight = std::sqrt(k*volume); // admm weight
}

void TetEnergyTerm::get_reduction( std::vector< Eigen::Triplet<double> > &triplets ){

	Matrix<double,4,3> S; // Selector
	S.setZero();
	S(0,0) = -1;	S(0,1) = -1;	S(0,2) = -1;
	S(1,0) =  1;
			S(2,1) =  1;
					S(3,2) =  1;
	Eigen::Matrix<double,4,3> D = S * edges_inv;
	Eigen::Matrix<double,3,4> Dt = D.transpose(); // Reduction

	const int rows[3] = { 0, 3, 6 };
	const int cols[4] = { 3*tet[0], 3*tet[1], 3*tet[2], 3*tet[3] };
	for( int r=0; r<3; ++r ){
		for( int c=0; c<4; ++c ){
			double value = Dt(r,c);
			for( int j=0; j<3; ++j ){
				triplets.emplace_back( rows[r]+j, cols[c]+j, value );
			}
		}
	}
}

void TetEnergyTerm::prox( VecX &zi ){
	typedef Matrix<double,9,1> Vector9d;
	Matrix<double,3,3> F = Map<Matrix<double,3,3> >(zi.data());
	JacobiSVD< Matrix<double,3,3> > svd(F, ComputeFullU | ComputeFullV);
	Vec3 S = Vec3::Ones();
	// Flip last singular value if inverted
	if( F.determinant() < 0.f ){ S[2] = -1.0; }
	// Project onto constraint
	Matrix<double,3,3> proj = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();
	Vector9d p = Map<Vector9d>(proj.data());
	// Update zi
	zi = 0.5 * ( p + zi );
}

double TetEnergyTerm::energy( const VecX &vecF ) {
	typedef Matrix<double,9,1> Vector9d;
	Vector9d vecF_ = vecF; // data() function is non-const
	Matrix<double,3,3> F = Map<Matrix<double,3,3> >(vecF_.data());
	JacobiSVD< Matrix<double,3,3> > svd(F, ComputeFullU | ComputeFullV);
	Vec3 S = Vec3::Ones();
	// Flip last singular value if inverted
	if( F.determinant() < 0.f ){ S[2] = -1.0; }
	// Project onto constraint
	Matrix<double,3,3> P = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();
	double k = weight*weight;  // projective dynamics weight
	double energy = k/2.0 * ( F - P ).squaredNorm();
	return energy;
}

double TetEnergyTerm::gradient( const VecX &F, VecX &grad ) {
	(void)(F);
	(void)(grad);
	throw std::runtime_error("**TetEnergyTerm TODO: gradient function");
	return 0;
}

//
//	NeoHookean
//

double NeoHookeanTet::NHProx::energy_density(const Vec3 &x) const {
	double J = x[0]*x[1]*x[2];
	double I_1 = x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
	double I_3 = J*J;
	double log_I3 = std::log( I_3 );
	double t1 = 0.5 * mu * ( I_1 - log_I3 - 3.0 );
	double t2 = 0.125 * lambda * log_I3 * log_I3;
	double r = t1 + t2;
	return r;
}

double NeoHookeanTet::NHProx::value(const Vec3 &x){
	if( x[0]<0.0 || x[1]<0.0 || x[2]<0.0 ){
		// No Mr. Linesearch, you have gone too far!
		return std::numeric_limits<float>::max();
	}
	double t1 = energy_density(x); // U(Dx)
	double t2 = (k*0.5) * (x-x0).squaredNorm(); // quad penalty
	return t1 + t2;
}


double NeoHookeanTet::NHProx::gradient(const Vec3 &x, Vec3 &grad){
	double J = x[0]*x[1]*x[2];
	if( J <= 0.0 ){
		throw std::runtime_error("NeoHookeanTet::NHProx::gradient Error: J <= 0");
	} else {
		Eigen::Vector3d x_inv(1.0/x[0],1.0/x[1],1.0/x[2]);
		grad = (mu * (x - x_inv) + lambda * std::log(J) * x_inv) + k*(x-x0);
	}
	return value(x);
}


NeoHookeanTet::NeoHookeanTet( const Vec4i &tet_, const std::vector<Eigen::Vector3d> &verts, const Lame &lame_ ) :
	TetEnergyTerm( tet_, verts, lame_ ){

	double k = lame.bulk_modulus();
	weight = std::sqrt(k*volume);

	problem.mu = lame.mu;
	problem.lambda = lame.lambda;
	problem.k = k;
}

void NeoHookeanTet::prox( VecX &zi ) {

	typedef Matrix<double,9,1> Vec9;
	typedef Matrix<double,3,3> Mat3;

	Mat3 F = Map<Mat3>(zi.data());
	Vec3 S; Mat3 U, V;
	signed_svd( F, S, U, V );
	problem.x0 = S;

	// If everything is very low, It is collapsed to a point and the minimize
	// will likely fail. So we'll just inflate it a bit.
	const double eps = 1e-6;
	if( std::abs(S[0]) < eps && std::abs(S[1]) < eps && std::abs(S[2]) < eps ){
		S[0] = eps; S[1] = eps; S[2] = eps;
	}

	if( S[2] < 0.0 ){ S[2] = -S[2]; }
	solver.minimize( problem, S );

	Mat3 matp = U * S.asDiagonal() * V.transpose();
	zi = Map<Vec9>(matp.data());

}

double NeoHookeanTet::energy( const VecX &F_ ) {
	Matrix<double,9,1> Fcopy = F_;
	Matrix<double,3,3> F = Map<Matrix<double,3,3> >(Fcopy.data());
	Vec3 S; Mat3 U, Vt;
	signed_svd( F, S, U, Vt );
	problem.x0 = S;
	if( S[2] < 0.0 ){ S[2] = -S[2]; }
	return problem.value(S)*volume;
}

double NeoHookeanTet::gradient( const VecX &F_, VecX &grad ) {
	Matrix<double,9,1> Fcopy = F_;
	Matrix<double,3,3> F = Map<Matrix<double,3,3> >(Fcopy.data());
	if( grad.rows() != 3 ){
		throw std::runtime_error("**NeoHookeanTet::gradient Error: grad not dim 3");
	}
	Vec3 S; Mat3 U, Vt;
	signed_svd( F, S, U, Vt );
	problem.x0 = S;
	if( S[2] < 0.0 ){ S[2] = -S[2]; }
	Vec3 g = grad;
	double e = problem.gradient(S,g)*volume;
	grad = g;
	return e;
}




//
//	St Venant-Kirchhoff
//

double StVKTet::StVKProx::value(const Vec3 &x){
	if( x[0]<0.0 || x[1]<0.0 || x[2]<0.0 ){
		// No Mr. Linesearch, you have gone too far!
		return std::numeric_limits<float>::max();
	}
	double t1 = energy_density(x); // U(Dx)
	double t2 = (k*0.5) * (x-x0).squaredNorm(); // quad penalty
	return t1 + t2;
}

double StVKTet::StVKProx::energy_density(const Vec3 &x) const {
	Vec3 x2( x[0]*x[0], x[1]*x[1], x[2]*x[2] );
	Vec3 st = 0.5 * ( x2 - Vec3(1,1,1) ); // strain tensor
	double st_tr2 = v3trace(st)*v3trace(st);
	double r = ( mu * ddot( st, st ) + ( lambda * 0.5 * st_tr2 ) );
	return r;
}


double StVKTet::StVKProx::gradient(const Vec3 &x, Vec3 &grad){
	Vec3 term1(
		mu * x[0]*(x[0]*x[0] - 1.0),
		mu * x[1]*(x[1]*x[1] - 1.0),
		mu * x[2]*(x[2]*x[2] - 1.0)
	);
	Vec3 term2 = 0.5 * lambda * ( x.dot(x) - 3.0 ) * x;
	grad = term1 + term2 + k*(x-x0);
	return value(x);
}

StVKTet::StVKTet( const Vec4i &tet_, const std::vector<Eigen::Vector3d> &verts, const Lame &lame_ ) :
	TetEnergyTerm( tet_, verts, lame_ ){

	double k = lame.bulk_modulus();
	weight = std::sqrt(k*volume);

	problem.mu = lame.mu;
	problem.lambda = lame.lambda;
	problem.k = k;
}

void StVKTet::prox( VecX &zi ) {

	typedef Matrix<double,9,1> Vec9;
	typedef Matrix<double,3,3> Mat3;

	Mat3 F = Map<Mat3>(zi.data());
	Vec3 S; Mat3 U, V;
	signed_svd( F, S, U, V );
	problem.x0 = S;

	// If everything is very low, It is collapsed to a point and the minimize
	// will likely fail. So we'll just inflate it a bit.
	const double eps = 1e-6;
	if( std::abs(S[0]) < eps && std::abs(S[1]) < eps && std::abs(S[2]) < eps ){
		S[0] = eps; S[1] = eps; S[2] = eps;
	}

	if( S[2] < 0.0 ){ S[2] = -S[2]; }

	solver.minimize( problem, S );
	Mat3 matp = U * S.asDiagonal() * V.transpose();
	zi = Map<Vec9>(matp.data());

}

double StVKTet::energy( const VecX &F_ ) {
	Matrix<double,9,1> Fcopy = F_;
	Matrix<double,3,3> F = Map<Matrix<double,3,3> >(Fcopy.data());
	Vec3 S; Mat3 U, Vt;
	signed_svd( F, S, U, Vt );
	problem.x0 = S;
	if( S[2] < 0.0 ){ S[2] = -S[2]; }
	return problem.value(S)*volume;
}

double StVKTet::gradient( const VecX &F_, VecX &grad ) {
	Matrix<double,9,1> Fcopy = F_;
	Matrix<double,3,3> F = Map<Matrix<double,3,3> >(Fcopy.data());
	if( grad.rows() != 3 ){
		throw std::runtime_error("**StVKTet::gradient Error: grad not dim 3");
	}
	Vec3 S; Mat3 U, Vt;
	signed_svd( F, S, U, Vt );
	problem.x0 = S;
	if( S[2] < 0.0 ){ S[2] = -S[2]; }
	Vec3 g = grad;
	double e = problem.gradient(S,g)*volume;
	grad = g;
	return e;
}


