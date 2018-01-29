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

#ifndef ADMM_CONSTRAINTSET_H
#define ADMM_CONSTRAINTSET_H

#include "Collider.hpp"
#include <unordered_map>

namespace admm {

class ConstraintSet {
public:
	typedef Eigen::Matrix<double,3,1> Vec3;
	typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;
	typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SparseMat;

	double constraint_w; // constraint stiffness (1 with hard constraints)
	std::shared_ptr<Collider> collider;
	std::unordered_map<int,Vec3> pins; // index -> location

	ConstraintSet() : constraint_w(1.0) {
		collider = std::make_shared<Collider>(Collider());
	}

	// Create a (equality) contraint matrix with
	// option of what to include from the set of constraints.
	// Collisions are created from passive_hits and dynamic_hits, so call
	// collider->detect to update collisions.
	inline void make_matrix( int dof, bool add_passive_collisions, bool add_dynamic_collisions );

	// Return the contraint matrix and solution vector. If dof differs from the cached
	// matrix (m_C, m_c) then an empty constraint matrix is created.
	inline void get_matrix( int dof, SparseMat &C, VecX &c );

private:
	SparseMat m_C;
	VecX m_c;

}; // end class linear solver


//
// Implementation
//


inline void ConstraintSet::get_matrix( int dof, SparseMat &C, VecX &c ){
	if( m_C.cols() != dof ){
		m_C.resize(1,dof);
		m_c = VecX::Zero(1);
	}
	C = m_C;
	c = m_c;
}


inline void ConstraintSet::make_matrix( int dof, bool add_passive_collisions, bool add_dynamic_collisions ){

	bool avoid_duplicate = true;
	bool row_per_node = false;

	// Get constraint info
	int n_p_hits = add_passive_collisions ? collider->passive_hits.size() : 0;
	int n_d_hits = add_dynamic_collisions ? collider->dynamic_hits.size() : 0;
	double ck = std::max(0.0,constraint_w);
	int c_rows = n_p_hits + n_d_hits;
	if( row_per_node ){ c_rows = dof/3; }
	std::vector<double> constrained( dof/3, 0.f );

	m_c = VecX::Zero(c_rows);
	std::vector< Eigen::Triplet<double> > triplets;
	triplets.reserve( n_p_hits*3 + n_d_hits*12 );

	// Passive collisions:
	for( int i=0; i<n_p_hits; ++i ){
		const PassiveCollision::Payload *h = &collider->passive_hits[i];
		if( constrained[h->vert_idx] ){ continue; }
		if( avoid_duplicate && h->dx < constrained[h->vert_idx] ){
			constrained[h->vert_idx] = h->dx;
		}

		int ci = i;
		if( row_per_node ){ ci = h->vert_idx; }
		m_c[ci] = ck*h->normal.dot(h->point);
		triplets.emplace_back( ci, h->vert_idx*3+0, ck*h->normal[0] );
		triplets.emplace_back( ci, h->vert_idx*3+1, ck*h->normal[1] );
		triplets.emplace_back( ci, h->vert_idx*3+2, ck*h->normal[2] );
	}

	// dynamic collisions:
	for( int i=0; i<n_d_hits; ++i ){
		const DynamicCollision::Payload *h = &collider->dynamic_hits[i];
		if( constrained[h->vert_idx] ){ continue; }
		if( avoid_duplicate && h->dx < constrained[h->vert_idx] ){
			constrained[h->vert_idx] = h->dx;
		}

		int ci = i+n_p_hits;
		if( row_per_node ){ ci = h->vert_idx; }
		triplets.emplace_back( ci, h->vert_idx*3+0, ck*h->normal[0] );
		triplets.emplace_back( ci, h->vert_idx*3+1, ck*h->normal[1] );
		triplets.emplace_back( ci, h->vert_idx*3+2, ck*h->normal[2] );
		for(int j=0; j<3; ++j){
			triplets.emplace_back( ci, h->face[j]*3+0, -ck*h->normal[0]*h->barys[j] );
			triplets.emplace_back( ci, h->face[j]*3+1, -ck*h->normal[1]*h->barys[j] );
			triplets.emplace_back( ci, h->face[j]*3+2, -ck*h->normal[2]*h->barys[j] );
		}
	}

	m_C.resize( c_rows, dof );
	m_C.setFromTriplets( triplets.begin(), triplets.end() );
}


} // ns admm

#endif
