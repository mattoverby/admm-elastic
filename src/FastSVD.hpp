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


//
// TODO: An implementation of fast svd
//


#ifndef ADMM_FASTSVD_H
#define ADMM_FASTSVD_H 1

#include <Eigen/Dense>

// Relevent papers:
// Computing the Singular Value Decomposition of 3x3 matrices with minimal branching and elementary floating point operations, McAdams et al.
// Energetically Consistent Invertible Elasticity, Stomakhin et al.
// Invertible Finite Elements For Robust Simulation of Large Deformation, Irving et al.

namespace admm {
	namespace fsvd {
		template <typename T> using Vec3 = Eigen::Matrix<T,3,1>;
		template <typename T> using Mat3 = Eigen::Matrix<T,3,3>;
	}

	// Projection, Singular Values, SVD's U, SVD's V
	template <typename T>
	static inline void signed_svd( const fsvd::Mat3<T> &F, fsvd::Vec3<T> &S, fsvd::Mat3<T> &U, fsvd::Mat3<T> &V ){
		using namespace Eigen;

		JacobiSVD< fsvd::Mat3<T> > svd( F, ComputeFullU | ComputeFullV );
		S = svd.singularValues();
		U = svd.matrixU();
		V = svd.matrixV();
		fsvd::Mat3<T> J = Matrix3d::Identity();
		J(2,2) = -1.0;

		// Check for inversion: U
		if( U.determinant() < 0.0 ){
			U = U * J;
			S[2] = -S[2];
		}

		// Check for inversion: V
		if( V.determinant() < 0.0 ){
			fsvd::Mat3<T> Vt = V.transpose();
			Vt = J * Vt;
			V = Vt.transpose();
			S[2] = -S[2];
		}

	} // end signed svd

} // endns admm

#endif
