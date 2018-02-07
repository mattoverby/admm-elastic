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

#include "ExplicitForce.hpp"

using namespace admm;
using namespace Eigen;

//
//	Helper Functions
//
namespace helper {
	// x is the list of all positions in the scene, idxN are the scaled indices that make up the face.
	// Node that the returned n is not normalized so the area can be obtained as (0.5*n.norm()).
	static void triangle_norm( const Eigen::VectorXd &x, int idx0, int idx1, int idx2, Eigen::Vector3d &normal, Eigen::Vector3d &tangent ){
		using namespace Eigen;
		Vector3d p0( x[idx0+0], x[idx0+1], x[idx0+2] );
		Vector3d p1( x[idx1+0], x[idx1+1], x[idx1+2] );
		Vector3d p2( x[idx2+0], x[idx2+1], x[idx2+2] );
		Vector3d a( p1-p0 );
		Vector3d b( p2-p0 );
		normal = ( a.cross(b) );
		tangent = a;
	}
}

//
//	Explicit Force
//

void WindForce::project( double dt, VectorXd &x, VectorXd &v, VectorXd &m ) const {
	(void)(m);

	// ANIMATION AERODYNAMICS (1991)
	// Wejchert and Haumann

	int n_tris = tris.size()/3;

	#pragma omp parallel for
	for( int i=0; i<n_tris; ++i ){

		int idx[3] = { tris[i*3+0]*3, tris[i*3+1]*3, tris[i*3+2]*3 };

		// Current velocity
		Vector3d curr_v = Vector3d( 
			v[idx[0]+0]+v[idx[1]+0]+v[idx[2]+0],
			v[idx[0]+1]+v[idx[1]+1]+v[idx[2]+1],
			v[idx[0]+2]+v[idx[1]+2]+v[idx[2]+2]
		) / 3.0;

		// Relative velocity
		Vector3d v_r = curr_v - direction;

		// Triangle normal
		Vector3d n, t;
		helper::triangle_norm( x, idx[0], idx[1], idx[2], n, t );
		Vector3d normal = n.normalized();
//		Vector3d tangent = t.normalized();

		// Other parameters of the wind force
		double area = 0.5 * n.norm();
		double alpha_n = 1000.0; // coupling strenth
//		double alpha_t = 1000.0; // coupling strenth

		// Normal force
		double v_n = normal.dot(v_r);
		Vector3d force_n = -alpha_n * area * v_n * fabs(v_n) * normal;

		// Tangent force
//		double v_t = fabs( tangent.dot( -(direction.normalized()) ) );
//		Vector3d force_t = alpha_t * area * direction * v_t;

		// Scale force before adding it to three nodes
		Vector3d force = force_n;// + force_t;
		force *= 0.33;
		force *= dt;

		// Ew, locks. Should reform this loop better
		#pragma omp critical
		for( int j=0; j<3; ++j ){
			v[ idx[j]+0 ] += force[0];
			v[ idx[j]+1 ] += force[1];
			v[ idx[j]+2 ] += force[2];
		}

	}

} // end wind force update





