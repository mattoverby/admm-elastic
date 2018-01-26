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

#include "Application.hpp"
#include "MCL/MeshIO.hpp"

using namespace mcl;


int main(int argc, char **argv){
/*
	// Load the mesh
	std::stringstream spherefile;
	spherefile << ADMMELASTIC_ROOT_DIR << "/samples/data/sphere";
	std::vector< mcl::TetMesh::Ptr > meshes = { mcl::TetMesh::create() };
	mcl::meshio::load_elenode( meshes[0].get(), spherefile.str() );

	Application app;
	if( app.parse_args( argc, argv ) ){ return EXIT_SUCCESS; }
	admm::Lame very_soft_rubber(1000000,0.299);
	app.add_dynamic_meshes( meshes, very_soft_rubber );

	// Add a collision floor
	float floor_y = -1.f;
	app.solver->m_collider.add_passive_obj(
		std::shared_ptr<admm::PassiveCollision>(new admm::Floor(floor_y))
	);

	// Add a floor renderable
	std::shared_ptr<mcl::TriangleMesh> floor = mcl::factory::make_plane(2,2);
	mcl::XForm<float> xf = mcl::xform::make_trans(0.f,floor_y,0.f) *
		mcl::xform::make_rot(-90.f,mcl::Vec3f(1,0,0)) *
		mcl::xform::make_scale(4.f,4.f,4.f);
	floor->apply_xform(xf);
	std::vector< mcl::TriangleMesh::Ptr > static_meshes = {floor};

	// Add the dynamic mesh, but remove it as a collider to avoid self collision
	app.add_static_meshes( static_meshes );
	app.solver->m_collider.dynamic_objs.clear();

	// Sim settings
	app.solver->settings.admm_iters = 100;

	bool success = app.display();

	if( !success ){ return EXIT_FAILURE; }
*/
	return EXIT_SUCCESS;
}

