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

using namespace admm;

Application app;
bool single_point_init = false;
void set_vertices();

int main(int argc, char *argv[]){

	if( argc > 1 ){
		std::string arg(argv[1]); 
		if( arg=="point" ){ single_point_init = true; }
		if( arg=="rand" ){ single_point_init = false; }
	}

	// Load the mesh
	std::stringstream tetfile;
	tetfile << ADMMELASTIC_ROOT_DIR << "/samples/data/bunny_1124";
	mcl::TetMesh::Ptr mesh = mcl::TetMesh::create();
	mcl::meshio::load_elenode( mesh.get(), tetfile.str() );
	mesh->flags |= binding::NOSELFCOLLISION | binding::STVK;
//	mesh->flags |= binding::NOSELFCOLLISION | binding::LINEAR;

	mcl::XForm<float> scale = mcl::xform::make_scale<float>(10.f,10.f,10.f);
	mcl::XForm<float> rotate = mcl::xform::make_rot<float>(20.f,mcl::Vec3f(1,0,0));
	mesh->apply_xform(rotate*scale);

	app.add_dynamic_mesh( mesh );
	app.renderWindow->m_camera->fov_deg() = 30.f; // zoom

	// Try to init the solver
	admm::Solver::Settings settings;
	if( settings.parse_args(argc,argv) ){ return EXIT_SUCCESS; }
	settings.linsolver = 0; // LDLT
	settings.gravity = 0;
	if( !app.solver->initialize(settings) ){ return EXIT_FAILURE; }

	// Create opengl context
	GLFWwindow* window = app.renderWindow->init();
	if( !window ){ return EXIT_FAILURE; }

	// Add render meshes
	int n_d_meshes = app.dynamic_meshes.size();
	for( int i=0; i<n_d_meshes; ++i ){
		app.renderWindow->add_mesh( app.dynamic_meshes[i].surface );
	}

	// Set the vertices (point/rand)
	set_vertices();
	app.dynamic_meshes[0].update( app.solver.get() );

	// Game loop
	bool runounce = true;
	while( app.renderWindow->is_open() ){

		//
		//	Update
		//
		if( app.controller->sim_running ){
			app.solver->step();
			for( int i=0; i<n_d_meshes; ++i ){ app.dynamic_meshes[i].update( app.solver.get() ); }
			if( single_point_init && runounce ){
				app.renderWindow->m_camera->eye() = mcl::Vec3f(0,0,10);
				runounce = false;
			}
		} // end run continuously
		else if( app.controller->sim_dostep ){
			app.controller->sim_dostep = false;
			app.solver->step();
			for( int i=0; i<n_d_meshes; ++i ){ app.dynamic_meshes[i].update( app.solver.get() ); }
			if( single_point_init && runounce ){
				app.renderWindow->m_camera->eye() = mcl::Vec3f(0,0,10);
				runounce = false;
			}
		} // end do one step

		//
		//	Render
		//
		app.renderWindow->draw();
		glfwPollEvents();

	} // end game loop

	return EXIT_SUCCESS;
}


std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dis(-0.75,0.75);
void set_vertices(){
	int dof = app.solver->m_x.rows();
	// Move the nodes of the vertices to a bad location
	if( single_point_init ){
		for( int i=0; i<dof; i+=3 ){
			app.solver->m_x[i] = 0.0;
			app.solver->m_x[i+1] = 0.0;
			app.solver->m_x[i+2] = 0.0;
		}
	} else {
		for( int i=0; i<dof; i+=3 ){
			app.solver->m_x[i]   = dis(gen);
			app.solver->m_x[i+1] = dis(gen);
			app.solver->m_x[i+2] = dis(gen);
		}
	}
}


