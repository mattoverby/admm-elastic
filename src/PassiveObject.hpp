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


#ifndef ADMM_PASSIVECOLLISION_HPP
#define ADMM_PASSIVECOLLISION_HPP

#include <Eigen/Dense>
#include "Collider.hpp"
#include "MCL/TetMesh.hpp"
#include "MCL/TriangleMesh.hpp"
#include "MCL/BVH.hpp"

namespace admm {

class Floor : public PassiveCollision {
public:
	typedef Eigen::Matrix<double,3,1> Vec3;

	double m_y;
	Floor( double y ) : m_y(y) {}
	void signed_distance( const Vec3 &x, Payload &p ) const {
		double dx = ( x[1] - m_y );
		if( dx > p.dx ){ return; }
		p.dx = dx;
		p.point = Vec3(x[0],m_y,x[2]);
		p.normal = Vec3(0,1,0);
	}
};


class Sphere : public PassiveCollision {
public:
	typedef Eigen::Matrix<double,3,1> Vec3;

	Vec3 center;
	double rad;
	Sphere( const Vec3 &c, double r ) : center(c), rad(r) {}
	void signed_distance( const Vec3 &x, Payload &p ) const {
		Vec3 dir = x - center;
		double dx = dir.norm() - rad;
		if( dx > p.dx ){ return; }
		dir.normalize();
		p.dx = dx;
		p.point = center + dir*rad;
		p.normal = dir;
	}
};


class PassiveMesh : public PassiveCollision {
public:
	typedef Eigen::Matrix<double,3,1> Vec3;
	mcl::bvh::AABBTree<float,3> tri_tree;
	mcl::bvh::AABBTree<float,4> tet_tree;

	std::shared_ptr<mcl::TetMesh> mesh;
	PassiveMesh( std::shared_ptr<mcl::TetMesh> mesh_ ) : mesh(mesh_){ update_bvh(); }

	void update_bvh(){
		mesh->need_faces();
		tri_tree.init( &mesh->faces[0][0], &mesh->vertices[0][0], mesh->faces.size() );
		tet_tree.init( &mesh->tets[0][0], &mesh->vertices[0][0], mesh->tets.size() );
	}

	void signed_distance( const Vec3 &x, Payload &p ) const {

		// First, check if objet is inside mesh
		mcl::bvh::PointInTet<float> p_in_mesh( x.cast<float>(), &mesh->vertices[0][0], &mesh->tets[0][0] );
		bool hit = tet_tree.traverse( p_in_mesh );

		// If there is an odd number of intersections, we are inside the mesh
		if( hit ){
			mcl::bvh::NearestTriangle<float> nearest_tri( x.cast<float>(), &mesh->vertices[0][0], &mesh->faces[0][0] );
			tri_tree.traverse( nearest_tri );

			mcl::Vec3i hit_face = mesh->faces[ nearest_tri.hit_tri ];
			const mcl::Vec3f &p0 = mesh->vertices[hit_face[0]];
			const mcl::Vec3f &p1 = mesh->vertices[hit_face[1]];
			const mcl::Vec3f &p2 = mesh->vertices[hit_face[2]];
			mcl::Vec3f norm = (p1-p0).cross(p2-p0);
			norm.normalize();

			// Set the payload data
			p.dx = -1.0*(nearest_tri.proj.cast<double>()-x).norm();
			p.point = nearest_tri.proj.cast<double>();
			p.normal = norm.cast<double>();
		}

	} // end signed distance
};


/*
class PassiveMesh : public PassiveCollision {
public:
	typedef Eigen::Matrix<double,3,1> Vec3;
	mcl::bvh::AABBTree<float,3> m_tree;

	std::shared_ptr<mcl::TriangleMesh> mesh;
	PassiveMesh( std::shared_ptr<mcl::TriangleMesh> mesh_ ) : mesh(mesh_){ update_bvh(); }

	void update_bvh(){
		m_tree.init( &mesh->faces[0][0], &mesh->vertices[0][0], mesh->faces.size() );
	}

	void signed_distance( const Vec3 &x, Payload &p ) const {

		// First, check if objet is inside mesh
		mcl::bvh::RayMultiHit<float> p_in_mesh( x.cast<float>(), &mesh->vertices[0][0], &mesh->faces[0][0] );
		m_tree.traverse( p_in_mesh );

		// If there is an odd number of intersections, we are inside the mesh
		if( p_in_mesh.hit_count % 2 == 1 ){
			mcl::bvh::NearestTriangle<float> nearest_tri( x.cast<float>(), &mesh->vertices[0][0], &mesh->faces[0][0] );
			m_tree.traverse( nearest_tri );

			mcl::Vec3i hit_face = mesh->faces[ nearest_tri.hit_tri ];
			const mcl::Vec3f &p0 = mesh->vertices[hit_face[0]];
			const mcl::Vec3f &p1 = mesh->vertices[hit_face[1]];
			const mcl::Vec3f &p2 = mesh->vertices[hit_face[2]];
			mcl::Vec3f norm = (p1-p0).cross(p2-p0);
			norm.normalize();

			// Set the payload data
			p.dx = -1.0*(nearest_tri.proj.cast<double>()-x).norm();
			p.point = nearest_tri.proj.cast<double>();
			p.normal = norm.cast<double>();
		}

	} // end signed distance
};
*/


} // end of namespace admm

#endif
