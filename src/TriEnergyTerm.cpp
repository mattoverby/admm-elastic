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

#include "TriEnergyTerm.hpp"
#include <iostream>

using namespace admm;

//
//	TriEnergyTerm
//

TriEnergyTerm::TriEnergyTerm( const Vec3i &tri_, const std::vector<Vec3> &verts, const Lame &lame_ ) :
	tri(tri_), lame(lame_), area(0.0), weight(0.0) {

	if( lame.limit_min > 1.0 ){ throw std::runtime_error("**TriEnergyTerm Error: Strain limit min should be -inf to 1"); }
	if( lame.limit_max < 1.0 ){ throw std::runtime_error("**TriEnergyTerm Error: Strain limit max should be 1 to inf"); }
	Vec3 e12 = verts[1] - verts[0];
	Vec3 e13 = verts[2] - verts[0];
	Vec3 n1 = e12.normalized();
	Vec3 n2 = (e13 - e13.dot(n1)*n1).normalized();
	Eigen::Matrix<double,3,2> basis;
	Eigen::Matrix<double,3,2> edges;
	basis.col(0) = n1; basis.col(1) = n2;
	edges.col(0) = e12; edges.col(1) = e13;
	rest_pose = (basis.transpose() * edges).inverse(); // Rest pose matrix

	area = (basis.transpose() * edges).determinant() / 2.0f;
	if( area < 0 ){
		throw std::runtime_error("**TriEnergyTerm Error: Inverted initial pose");
	}

	double k = lame.bulk_modulus();
	weight = std::sqrt(k*area);
}


void TriEnergyTerm::get_reduction( std::vector< Eigen::Triplet<double> > &triplets ){

	Eigen::Matrix<double,3,2> S;
	S.setZero();
	S(0,0) = -1;	S(0,1) = -1;
	S(1,0) =  1;
			S(2,1) =  1;

	Eigen::Matrix<double,3,2> D = S * rest_pose;
	int cols[3] = { 3*tri[0], 3*tri[1], 3*tri[2] };
	for( int i=0; i<3; ++i ){
		for( int j=0; j<3; ++j ){
			triplets.emplace_back( i, cols[j]+i, D(j,0) );
			triplets.emplace_back( 3+i, cols[j]+i, D(j,1) );
		}
	}
}


void TriEnergyTerm::prox( VecX &zi ){
	using namespace Eigen;
	typedef Matrix<double,6,1> Vector6d;

	Matrix<double,3,2> F = Map<Matrix<double,3,2> >(zi.data());
	JacobiSVD<Matrix<double,3,2> > svd(F, ComputeFullU | ComputeFullV);
	Matrix<double,3,2> S = Matrix<double,3,2>::Zero();
	S.block<2,2>(0,0) = Matrix<double,2,2>::Identity();
	Matrix<double,3,2> P = svd.matrixU() * S * svd.matrixV().transpose();
	Vector6d p = Map<Vector6d>(P.data());
	zi = 0.5 * ( p + zi );

	// If w^2 != k*volume, use this:
//	double k = lame.bulk_modulus();
//	double ka = k * area;
//	double w2 = weight*weight;
//	zi = (ka*p + w2*zi) / (w2 + ka);

	const bool check_strain = lame.limit_min > 0.0 || lame.limit_max < 99.0;
	if( check_strain ){
		double l_col0 = zi.head<3>().norm();
		double l_col1 = zi.tail<3>().norm();
		if( l_col0 < lame.limit_min ){ zi.head<3>() *= ( lame.limit_min / l_col0 ); }
		if( l_col1 < lame.limit_min ){ zi.tail<3>() *= ( lame.limit_min / l_col1 ); }
		if( l_col0 > lame.limit_max ){ zi.head<3>() *= ( lame.limit_max / l_col0 ); }
		if( l_col1 > lame.limit_max ){ zi.tail<3>() *= ( lame.limit_max / l_col1 ); }
	}

}


double TriEnergyTerm::energy( const VecX &vecF ) {
	using namespace Eigen;
	VecX vecF_copy = vecF;
	Matrix<double,3,2> F = Map<Matrix<double,3,2> >(vecF_copy.data());
	JacobiSVD<Matrix<double,3,2> > svd(F, ComputeFullU | ComputeFullV);
	Matrix<double,3,2> S = Matrix<double,3,2>::Zero();
	S.block<2,2>(0,0) = Matrix<double,2,2>::Identity();
	Matrix<double,3,2> P = svd.matrixU() * S * svd.matrixV().transpose();
	double k = lame.bulk_modulus();
	return 0.5 * k * area * ( F - P ).squaredNorm();
}


double TriEnergyTerm::gradient( const VecX &vecF, VecX &grad ) {
	(void)(vecF);
	(void)(grad);
	throw std::runtime_error("**TriEnergyTerm TODO: gradient function");
	return 0;
}


