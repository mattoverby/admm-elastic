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

#include "Application.hpp"
#include <random>
#include "MCL/XForm.hpp"
#include "MCL/ShapeFactory.hpp"

using namespace admm;

Application app;
void get_pins( mcl::TriangleMesh::Ptr mesh, std::vector<int> &pin_ids, int idx_offset );

int main(int argc, char *argv[]){

	admm::Solver::Settings settings;
	if( settings.parse_args(argc,argv) ){ return EXIT_SUCCESS; }

	std::vector< mcl::TriangleMesh::Ptr > meshes = {
		mcl::factory::make_plane( 10, 10 ),
		mcl::factory::make_plane( 10, 10 )
	};

	meshes[0]->flags = binding::NOSELFCOLLISION | binding::LINEAR;
	meshes[1]->flags = binding::NOSELFCOLLISION | binding::LINEAR;
	const mcl::XForm<float> xf_left = mcl::xform::make_trans(-2.f, 0.f, 0.f);
	const mcl::XForm<float> xf_right = mcl::xform::make_trans(2.f, 0.f, 0.f);
	meshes[0]->apply_xform( xf_left );
	meshes[1]->apply_xform( xf_right );

	// Add meshes to the system
	admm::Lame very_soft_rubber(100,0.1);
	app.add_dynamic_mesh( meshes[1], very_soft_rubber );
	very_soft_rubber.limit_min = 0.95;
	very_soft_rubber.limit_max = 1.05;
	app.add_dynamic_mesh( meshes[0], very_soft_rubber );

	// Pin corners
	std::vector<int> pins;
	int pin_idx_offset = 0;
	for( int i=0; i<(int)meshes.size(); ++i ){
		get_pins( meshes[i], pins, pin_idx_offset );
		pin_idx_offset += meshes[i]->vertices.size();
	}
	app.solver->set_pins( pins );

	// Try to init the solver
	if( !app.solver->initialize(settings) ){ return EXIT_FAILURE; }

	// Create opengl context
	GLFWwindow* window = app.renderWindow->init();
	if( !window ){ return EXIT_FAILURE; }

	// Add render meshes
	int n_d_meshes = app.dynamic_meshes.size();
	for( int i=0; i<n_d_meshes; ++i ){
		app.renderWindow->add_mesh( app.dynamic_meshes[i].surface );
	}

	// Game loop
	while( app.renderWindow->is_open() ){

		//
		//	Update
		//
		if( app.controller->sim_running ){
			app.solver->step();
			for( int i=0; i<n_d_meshes; ++i ){ app.dynamic_meshes[i].update( app.solver.get() ); }
		} // end run continuously
		else if( app.controller->sim_dostep ){
			app.controller->sim_dostep = false;
			app.solver->step();
			for( int i=0; i<n_d_meshes; ++i ){ app.dynamic_meshes[i].update( app.solver.get() ); }
		} // end do one step

		//
		//	Render
		//
		app.renderWindow->draw();
		glfwPollEvents();

	} // end game loop

	return EXIT_SUCCESS;
}

void get_pins( mcl::TriangleMesh::Ptr mesh, std::vector<int> &pin_ids, int idx_offset ){

	Eigen::AlignedBox<float,3> aabb = mesh->bounds();

	int left_idx = -1; // min x
	int right_idx = -1; // max x
	float min_y = aabb.max()[1] - 1e-3f;
	float curr_max_x = -99999.f;
	float curr_min_x = -curr_max_x;
	
	int n_v = mesh->vertices.size();
	for( int i=0; i<n_v; ++i ){
		float y = mesh->vertices[i][1];
		if( y < min_y ){ continue; }

		float x = mesh->vertices[i][0];				
		if( x < curr_min_x ){
			left_idx = i;
			curr_min_x = x;
		}
		else if( x > curr_max_x ){
			right_idx = i;
			curr_max_x = x;
		}
	}

	if( left_idx < 0 || right_idx < 0 ){
		throw std::runtime_error("Failed to find pin locations");
	}

	pin_ids.emplace_back( left_idx + idx_offset );
	pin_ids.emplace_back( right_idx + idx_offset );

}

