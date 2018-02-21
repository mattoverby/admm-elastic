// Copyright (c) 2016 University of Minnesota
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


#ifndef ADMM_DYNAMICCOLLISION_HPP
#define ADMM_DYNAMICCOLLISION_HPP

#include <Eigen/Dense>
#include "MCL/BVH.hpp"
#include "Collider.hpp"

namespace admm {


class TetMeshCollision : public DynamicCollision {
private:
	mcl::bvh::AABBTree<float,3> m_faces_tree;
	mcl::bvh::AABBTree<double,4> m_tets_tree;

	int vert_offset;
	std::vector<mcl::Vec4i> mesh_tets;
	std::vector<mcl::Vec3i> mesh_faces;
	std::vector<mcl::Vec3f> mesh_rest_verts;
	const VecX *mesh_verts;

public:
	typedef Eigen::Matrix<double,3,1> Vec3;
	typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;

	// Constructor with vertex offset (to index into global vertex array)
	TetMeshCollision( const std::shared_ptr<mcl::TetMesh> mesh, int v_offset ) : vert_offset(v_offset) {
		if( mesh->faces.size() == 0 ){
			throw std::runtime_error( "**TetMeshCollision Error: TetMesh needs surface faces" );
		}

		// Create a tree of the rest faces
		mesh_faces = mesh->faces;
		mesh_rest_verts = mesh->vertices;
		m_faces_tree.init( &mesh_faces[0][0], &mesh->vertices[0][0], mesh_faces.size() );

		// Update tet indices
		int n_tets = mesh->tets.size();
		mesh_tets = mesh->tets;
		const mcl::Vec4i tet_offset = mcl::Vec4i(vert_offset,vert_offset,vert_offset,vert_offset);
		for( int i=0; i<n_tets; ++i ){ mesh_tets[i] += tet_offset; }

	} // end constructor

	// Update the dynamic tet mesh vertices
	void update( const VecX &x ){
		mesh_verts = &x;
		m_tets_tree.init( &mesh_tets[0][0], mesh_verts->data(), mesh_tets.size() );
	}

	// Compute signed distance on the tet mesh
	void signed_distance( const Vec3 &x, Payload &p ) const {
		if( p.dx < 0 ){ return; }// only resolve one dynamic collision at a time

		// Do a point-in-tet test
		mcl::bvh::PointInTet<double> point_in_tet( x, mesh_verts->data(), &mesh_tets[0][0] );
		point_in_tet.skip_vert_idx.push_back( p.vert_idx );
		bool hit = m_tets_tree.traverse( point_in_tet );

		// If we're inside a tet, find the nearest surface
		if( hit ){

			Vec3 restx;
			{ // Compute point in rest pose
				mcl::Vec4i tet = mesh_tets[ point_in_tet.hit_tet ];
				const Vec3 &p0 = mesh_verts->segment<3>(tet[0]*3);
				const Vec3 &p1 = mesh_verts->segment<3>(tet[1]*3);
				const Vec3 &p2 = mesh_verts->segment<3>(tet[2]*3);
				const Vec3 &p3 = mesh_verts->segment<3>(tet[3]*3);
				mcl::Vec4d hitbarys = mcl::vec::barycoords( x, p0, p1, p2, p3 );
				tet -= mcl::Vec4i(vert_offset,vert_offset,vert_offset,vert_offset);
				restx = hitbarys[0]*mesh_rest_verts[tet[0]].cast<double>() +
					hitbarys[1]*mesh_rest_verts[tet[1]].cast<double>() +
					hitbarys[2]*mesh_rest_verts[tet[2]].cast<double>() +
					hitbarys[3]*mesh_rest_verts[tet[3]].cast<double>();
			}

			mcl::bvh::NearestTriangle<float> nearest_tri( restx.cast<float>(), &mesh_rest_verts[0][0], &mesh_faces[0][0] );
			nearest_tri.skip_vert_idx.push_back( p.vert_idx-vert_offset );
			m_faces_tree.traverse( nearest_tri );
			if( nearest_tri.hit_tri < 0 ){
				throw std::runtime_error("TetMeshCollision Error: Could not find a nearest face");
			}
			mcl::Vec3i hit_face = mesh_faces[ nearest_tri.hit_tri ];
			const mcl::Vec3f &p0 = mesh_rest_verts[hit_face[0]];
			const mcl::Vec3f &p1 = mesh_rest_verts[hit_face[1]];
			const mcl::Vec3f &p2 = mesh_rest_verts[hit_face[2]];
			mcl::Vec3f norm = (p1-p0).cross(p2-p0);
			norm.normalize();

			// Set the payload data
			p.dx = -1.0*(nearest_tri.proj-restx.cast<float>()).norm();
			p.face = hit_face + mcl::Vec3i(vert_offset,vert_offset,vert_offset);
			p.barys = mcl::vec::barycoords(nearest_tri.proj, p0, p1, p2).cast<double>();
			p.normal = norm.cast<double>();

		}

	} // end signed distance

}; // class dynamic collision


} // namespace admm

#endif

