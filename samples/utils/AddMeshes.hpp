// Copyright (c) 2017 University of Minnesota
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
// By Matt Overby (http://www.mattoverby.net)


//
// Classes/functions used by admm-elastic samples
//

#ifndef ADMM_ADDMESHES_H
#define ADMM_ADDMESHES_H

#include "MCL/MeshIO.hpp"
#include "MCL/ShapeFactory.hpp"
#include "PassiveObject.hpp"
#include "DynamicObject.hpp"
#include "TetEnergyTerm.hpp"
#include "TriEnergyTerm.hpp"
#include "Solver.hpp"

//
//	TODO revise
//

// Glue code to couple admm-elastic with mclscene data types
namespace binding {

	// The following functions add a mesh to the solver, which creates nodes and a dynamic collision obstacle.
	// Defaults:
	//	Tets: Mass (rubber), Model (non-limited linear elastic)
	//	Triangle: Mass (polyester), Model (strain-limited linear elastic)

	static inline void add_tetmesh( admm::Solver *solver, std::shared_ptr<mcl::TetMesh> &mesh,
		const admm::Lame &lame = admm::Lame::rubber(), bool verbose=true );

	static inline void add_trimesh( admm::Solver *solver, std::shared_ptr<mcl::TriangleMesh> &mesh,
		const admm::Lame &lame = admm::Lame::rubber(), bool verbose=true );

	// Flags that can be added to the mesh->flags member
	enum MeshFlags {
		NOSELFCOLLISION = 1 << 1,
		LINEAR = 1 << 2, // default when mesh->flags==0
		NEOHOOKEAN = 1 << 3,
		STVK = 1 << 4,
	};
}

// Used for pinning portions of a mesh.
// Usage:
//	GrabbySphere gs( center, radius );
//	std::vector<int> pins;
//	gs.get_indices( solver.m_x, pins )
class GrabbySphere {
public:
	mcl::Vec3f c; // center
	float r; // radius
	std::shared_ptr<mcl::TriangleMesh> mesh;

	GrabbySphere( mcl::Vec3f c_, float r_ ) : c(c_), r(r_) {
		mesh = mcl::factory::make_sphere( c, r, 32 );
	}

	// Returns a list of indices that are inside the sphere
	void get_indices( const Eigen::VectorXd &x, std::vector<int> &inds ){
		int n_verts = x.size()/3;
		mcl::Vec3d cd = c.cast<double>();
		for( int i=0; i<n_verts; ++i ){
			float d = (x.segment<3>(i*3) - cd).norm();
			if( d < r ){ inds.push_back( i ); }
		}
	}
	
	
};

//
//	Implementation
//

static inline void binding::add_tetmesh( admm::Solver *solver, std::shared_ptr<mcl::TetMesh> &mesh, const admm::Lame &lame, bool verbose ){

	// Add vertices to the solver
	int num_tet_verts = mesh->vertices.size(); // tet verts
	int prev_tet_verts = solver->m_x.rows()/3;
	int num_tets = mesh->tets.size();

	// Use rubber for masses: 1522 kg/m^3
	std::vector<float> masses;
	mesh->weighted_masses( masses, 1522.f );

	int n_masses = masses.size();
	for( int i=0; i<n_masses; ++i ){
		if( masses[i] <= 0.f ){
			throw std::runtime_error("TetMesh Error: Zero mass");
		}
	}

	// Add nodes to the solver
	solver->m_x.conservativeResize( prev_tet_verts*3 + num_tet_verts*3 );
	solver->m_masses.conservativeResize( prev_tet_verts*3 + num_tet_verts*3 );
	for( int i=0; i<num_tet_verts; ++i ){
		int idx = i+prev_tet_verts;
		solver->m_x.segment<3>(idx*3) = mesh->vertices[i].cast<double>();
		solver->m_masses.segment<3>(idx*3) = Eigen::Vector3d(1,1,1)*masses[i];
	}

	// Add a dynamic collider
	if( !(mesh->flags & NOSELFCOLLISION) ){
		mesh->need_faces();
		std::shared_ptr<admm::TetMeshCollision> collision_mesh(
			new admm::TetMeshCollision(mesh, prev_tet_verts)
		);
		solver->add_dynamic_collider( collision_mesh );
		std::vector<int> surf_inds;
		mesh->surface_inds( surf_inds );
		int n_inds = surf_inds.size();
		for( int i=0; i<n_inds; ++i ){
			solver->surface_inds.emplace_back( surf_inds[i]+prev_tet_verts );
		}
	}

	// Add individual tet forces
	if( (mesh->flags & LINEAR) || (mesh->flags==0) ){
		admm::create_tets_from_mesh<float,admm::TetEnergyTerm>(
			solver->energyterms,
			&mesh->vertices[0][0],
			&mesh->tets[0][0],
			num_tets,
			lame,
			prev_tet_verts
		);
	} else if ( mesh->flags & NEOHOOKEAN ) {
		admm::create_tets_from_mesh<float,admm::NeoHookeanTet>(
			solver->energyterms,
			&mesh->vertices[0][0],
			&mesh->tets[0][0],
			num_tets,
			lame,
			prev_tet_verts
		);
	} else if ( mesh->flags & STVK ) {
		admm::create_tets_from_mesh<float,admm::StVKTet>(
			solver->energyterms,
			&mesh->vertices[0][0],
			&mesh->tets[0][0],
			num_tets,
			lame,
			prev_tet_verts
		);
	}

	if( verbose ){
		std::cout << "Added mesh: " << 
			"\n\tmass: " << std::accumulate(masses.begin(), masses.end(), 0.f) << "kg" <<
			"\n\tvertices: " << num_tet_verts <<
			"\n\ttets: " << num_tets <<
			"\n\t(total) verts: " << solver->m_x.size()/3 <<
		std::endl;
	}
}


static inline void binding::add_trimesh( admm::Solver *solver, std::shared_ptr<mcl::TriangleMesh> &mesh,
	const admm::Lame &lame, bool verbose ){

	// Add vertices to the solver
	int num_tri_verts = mesh->vertices.size(); // tet verts
	int prev_tri_verts = solver->m_x.rows()/3;
	int num_tris = mesh->faces.size();

	std::vector<float> masses;
	mesh->weighted_masses( masses, 1.f ); // TODO some mass based on real world values

	int n_masses = masses.size();
	for( int i=0; i<n_masses; ++i ){
		if( masses[i] <= 0.f ){
			throw std::runtime_error("TriMesh Error: Zero mass");
		}
	}

	// Add nodes to the solver
	solver->m_x.conservativeResize( prev_tri_verts*3 + num_tri_verts*3 );
	solver->m_masses.conservativeResize( prev_tri_verts*3 + num_tri_verts*3 );
	for( int i=0; i<num_tri_verts; ++i ){
		int idx = i+prev_tri_verts;
		solver->m_x.segment<3>(idx*3) = mesh->vertices[i].cast<double>();
		solver->m_masses.segment<3>(idx*3) = Eigen::Vector3d(1,1,1)*masses[i];
	}

	// Add a dynamic collider
	if( !(mesh->flags & NOSELFCOLLISION) ){
		std::cerr << "TODO: Add triangle mesh as collision object" << std::endl;
	}

	// Add individual tet forces
	if( (mesh->flags & LINEAR) || (mesh->flags==0) ){
		admm::create_tris_from_mesh<float,admm::TriEnergyTerm>(
			solver->energyterms,
			&mesh->vertices[0][0],
			&mesh->faces[0][0],
			num_tris,
			lame,
			prev_tri_verts
		);
	} else {
		throw std::runtime_error("**binding::add_trimesh Error: Unknown triangle mesh material type");
	}

	if( verbose ){
		std::cout << "Added mesh: " << 
			"\n\tmass: " << std::accumulate(masses.begin(), masses.end(), 0.f) << "kg" <<
			"\n\tvertices: " << num_tri_verts <<
			"\n\ttris: " << num_tris <<
			"\n\t(total) verts: " << solver->m_x.size()/3 <<
		std::endl;
	}

}

#endif
