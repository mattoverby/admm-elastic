// Copyright (c) 2017 University of Minnesota
// 
// MCLSCENE Uses the BSD 2-Clause License (http://www.opensource.org/licenses/BSD-2-Clause)
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

	admm::Solver::Settings settings;
	settings.linsolver = 1;
	settings.admm_iters = 30;
	if( settings.parse_args( argc, argv ) ){ return EXIT_SUCCESS; }
	Application app(settings);

	std::vector< mcl::TetMesh::Ptr > meshes = {
		mcl::TetMesh::create(),
		mcl::TetMesh::create()
	};

	// Add meshes
	std::stringstream torusfile;
	torusfile << ADMMELASTIC_ROOT_DIR << "/samples/data/torus";
	int n_meshes = meshes.size();
	float rad = 0.f;
	for( int i=0; i<n_meshes; ++i ){
		mcl::meshio::load_elenode( meshes[i].get(), torusfile.str() );
		mcl::XForm<float> xf = mcl::xform::make_scale(0.2f,0.2f,0.2f);
		meshes[i]->apply_xform( xf );
		rad = ( meshes[i]->bounds().max() - meshes[i]->bounds().center() )[1];//*0.7f;
	}

	// Now move around the tori to link them as a chain
	for( int i=0; i<n_meshes; ++i ){
		mcl::XForm<float> rxf = mcl::xform::make_rot(i*90.f,Vec3f(0,1,0));
		mcl::XForm<float> txf = mcl::xform::make_trans(0.f, -i*rad, 0.f);
		mcl::XForm<float> swingxf = mcl::xform::make_rot(45.f,Vec3f(0,0,-1));
		meshes[i]->apply_xform( swingxf*rxf*txf );
		app.add_dynamic_mesh( meshes[i] );
	}

	// Create a grabby sphere that pins the top torus
	GrabbySphere sphere( Vec3f(0,0.25,0), rad*0.3 );

	std::vector<int> pins;
	sphere.get_indices( app.solver->m_x, pins );
	app.solver->set_pins( pins );
	app.add_static_mesh( sphere.mesh );

	// Set the camera to a nice spot
	app.renderWindow->m_camera->eye() = mcl::Vec3f(2,-0.55,2.5);
	app.renderWindow->m_camera->lookat() = mcl::Vec3f(0,-0.55,0);

	bool success = app.display();
	if( !success ){ return EXIT_FAILURE; }
	return EXIT_SUCCESS;
}

