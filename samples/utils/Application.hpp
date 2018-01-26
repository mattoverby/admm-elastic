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
// A helper class for OpenGL/physics interopt.
// Kinda clunky but works for now
//

#ifndef ADMM_APPLICATION_H
#define ADMM_APPLICATION_H

#include "AddMeshes.hpp"
#include "MCL/RenderWindow.hpp"
#include "Solver.hpp"
#include <iomanip>

class AppController : public mcl::Controller {
public:
	AppController() : sim_running(false), sim_dostep(false), sim_realtime(false),
		save_single_ss(false), save_all_ss(false) {}

	bool sim_running; // run sim each frame
	bool sim_dostep; // run a single sim time step
	bool sim_realtime; // try to match sim dt and frame dt
	bool save_single_ss; 
	bool save_all_ss;

	void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
	inline void print_help() const; // Press H for help
};

class Application {
public:
	std::shared_ptr<admm::Solver> solver;
	std::shared_ptr<mcl::RenderWindow> renderWindow;
	std::shared_ptr<AppController> controller;
	admm::Solver::Settings settings;

	Application( const admm::Solver::Settings &settings = admm::Solver::Settings() );

	// Adds a deformable tet mesh
	inline void add_dynamic_mesh( mcl::TetMesh::Ptr mesh,
		const admm::Lame &lame = admm::Lame::rubber() );

	// Adds a deformable triangle mesh
	inline void add_dynamic_mesh( mcl::TriangleMesh::Ptr mesh,
		const admm::Lame &lame = admm::Lame::rubber() );

	// Obstacles aren't simulated, but act as collision objects.
	inline void add_obstacle( std::shared_ptr<admm::PassiveCollision> c,
		mcl::TriangleMesh::Ptr surf );

	// updates renderables and collision bvh
//	inline void update_passive_meshes();

	// Static meshes, they don't do anything except be rendered
	inline void add_static_mesh( mcl::TriangleMesh::Ptr mesh );

	// Returns success or failure
	inline bool display();

	// Per-Frame and/or sim step callback, called right before.
	// Use if you want to do things like update passive/static meshes.
	std::function<void(float screen_dt)> frame_cb;
	std::function<void()> sim_cb;

	// I know it's not efficient to make copies of shared pointers,
	// but it sure is convenient...
	struct DynamicMesh {
		DynamicMesh() : mesh_type(-1), solver_v(0) {}
		short mesh_type; // 0=tri, 1=tet
		int solver_v; // index into solver's vertices
		void update( admm::Solver *solver ); // solver verts -> mesh verts
		// Null unless used:
		std::shared_ptr<mcl::TetMesh> tetmesh;
		std::shared_ptr<mcl::TriangleMesh> trimesh;
		std::shared_ptr<mcl::RenderMesh> surface;
	};

	struct PassiveMesh {
		std::shared_ptr<mcl::RenderMesh> surface;
		std::shared_ptr<admm::PassiveCollision> collidermesh;
	};

	struct StaticMesh {
		std::shared_ptr<mcl::RenderMesh> surface;
	};

	// A list of all meshes in the scene
	std::vector<DynamicMesh> dynamic_meshes;
	std::vector<PassiveMesh> passive_meshes;
	std::vector<StaticMesh> static_meshes;

private:
	int ss_counter;
	bool initialized;
	inline std::string make_screenshot_fn( int &counter );

}; // end class application


//
//	Implementation
//

Application::Application( const admm::Solver::Settings &settings_ ) : settings(settings_), ss_counter(0), initialized(false) {
	solver = std::make_shared<admm::Solver>();
	renderWindow = std::make_shared<mcl::RenderWindow>();
	controller = std::make_shared<AppController>();
	renderWindow->set_controller( controller );
}

inline void Application::add_dynamic_mesh( mcl::TetMesh::Ptr mesh, const admm::Lame &lame ){
	mesh->need_normals();
	dynamic_meshes.emplace_back( DynamicMesh() );
	dynamic_meshes.back().mesh_type = 1;
	dynamic_meshes.back().solver_v = solver->m_x.rows()/3; // global vertex idx
	dynamic_meshes.back().tetmesh = mesh; // copy the ptr
	dynamic_meshes.back().surface = std::make_shared<mcl::RenderMesh>( mesh, mcl::RenderMesh::DYNAMIC );
	binding::add_tetmesh( solver.get(), mesh, lame, settings.verbose );
}


inline void Application::add_dynamic_mesh( mcl::TriangleMesh::Ptr mesh, const admm::Lame &lame ){
	mesh->need_normals();
	dynamic_meshes.emplace_back( DynamicMesh() );
	dynamic_meshes.back().mesh_type = 0;
	dynamic_meshes.back().solver_v = solver->m_x.rows()/3; // global vertex idx
	dynamic_meshes.back().trimesh = mesh; // copy the ptr
	dynamic_meshes.back().surface = std::make_shared<mcl::RenderMesh>( mesh, mcl::RenderMesh::DYNAMIC );
	binding::add_trimesh( solver.get(), mesh, lame, settings.verbose );
}


inline void Application::add_obstacle( std::shared_ptr<admm::PassiveCollision> c, mcl::TriangleMesh::Ptr surf ){
	passive_meshes.emplace_back( PassiveMesh() );
	passive_meshes.back().surface = std::make_shared<mcl::RenderMesh>(surf);
	passive_meshes.back().surface->phong = mcl::material::Phong::create( mcl::material::Preset::Gunmetal );
	passive_meshes.back().collidermesh = c;
	solver->add_obstacle(passive_meshes.back().collidermesh);
}


inline void Application::add_static_mesh( mcl::TriangleMesh::Ptr mesh ){
	static_meshes.emplace_back( StaticMesh() );
	static_meshes.back().surface = std::make_shared<mcl::RenderMesh>(mesh);
	static_meshes.back().surface->phong = mcl::material::Phong::create( mcl::material::Preset::Gunmetal );
}

/*
inline void Application::update_passive_meshes(){
	int n_meshes = passive_meshes.size();
	for( int i=0; i<n_meshes; ++i ){
		passive_meshes[i].collidermesh->update_bvh();
		if( initialized ){
			passive_meshes[i].surface->load_buffers();
		}
	}
}
*/
// Returns success or failure
inline bool Application::display(){

	// Try to init the solver
	if( !solver->initialize(settings) ){ return false; }

	// Create opengl context
	GLFWwindow* window = renderWindow->init();
	if( !window ){ return false; }

	initialized = true;
	bool has_frame_cb( frame_cb );
	bool has_sim_cb( sim_cb );

	// Add render meshes
	int n_d_meshes = dynamic_meshes.size();
	for( int i=0; i<n_d_meshes; ++i ){
		renderWindow->add_mesh( dynamic_meshes[i].surface );
		dynamic_meshes[i].update( solver.get() );
	}
	int n_p_meshes = passive_meshes.size();
	for( int i=0; i<n_p_meshes; ++i ){
		renderWindow->add_mesh( passive_meshes[i].surface );
	}
	int n_s_meshes = static_meshes.size();
	for( int i=0; i<n_s_meshes; ++i ){
		renderWindow->add_mesh( static_meshes[i].surface );
	}

	// Compute some nice info
	if( solver->settings().verbose>0 ){
		Eigen::AlignedBox<float,3> aabb = renderWindow->bounds();
		mcl::Vec3f diag = aabb.max()-aabb.min();
		std::cout << "Scene radius: " << diag.norm()/2 << std::endl;
	}

	// Game loop
	float t_old = glfwGetTime();
	while( renderWindow->is_open() ){

		//
		//	Update
		//
		float t = glfwGetTime();
		float screen_dt = t - t_old;
		t_old = t;
		if( has_frame_cb ){ frame_cb(screen_dt); }
		if( controller->sim_running ){
			if( controller->sim_realtime ){
				while( screen_dt > 0.0 ){
					if( has_sim_cb ){ sim_cb(); }
					solver->step();
					screen_dt -= solver->settings().timestep_s;
				}
			} else {
				if( has_sim_cb ){ sim_cb(); }
				solver->step();
			}
			for( int i=0; i<n_d_meshes; ++i ){ dynamic_meshes[i].update( solver.get() ); }
		} // end run continuously
		else if( controller->sim_dostep ){
			controller->sim_dostep = false;
			if( has_sim_cb ){ sim_cb(); }
			solver->step();
			for( int i=0; i<n_d_meshes; ++i ){ dynamic_meshes[i].update( solver.get() ); }
		} // end do one step

		//
		//	Render
		//
		renderWindow->draw();
		glfwPollEvents();

		// Save screenshot?
		if( controller->save_single_ss || controller->save_all_ss ){
			controller->save_single_ss = false;
			std::string fn = make_screenshot_fn(ss_counter);
			renderWindow->save_screenshot( fn );
		}

	} // end game loop

	return true;

} // end display


inline std::string Application::make_screenshot_fn( int &counter ){
	std::stringstream ss;
	ss << ADMMELASTIC_OUTPUT_DIR << "/" << std::setfill('0') << std::setw(5) << counter << ".png";
	counter++;
	return ss.str();
}

void Application::DynamicMesh::update( admm::Solver *solver ){

	if( mesh_type==0 ){ // triangle
		int nv = trimesh->vertices.size();
		for( int i=0; i<nv; ++i ){
			int idx = i+solver_v;
			trimesh->vertices[i] = solver->m_x.segment<3>(idx*3).cast<float>();
		}
		trimesh->need_normals(true);
	}

	else if( mesh_type == 1 ){ // tet
		int nv = tetmesh->vertices.size();
		for( int i=0; i<nv; ++i ){
			int idx = i+solver_v;
			tetmesh->vertices[i] = solver->m_x.segment<3>(idx*3).cast<float>();
		}
		tetmesh->need_normals(true);
	}
	else{
		throw std::runtime_error("**DynamicMesh::update Error: Unknown mesh type");
	}

	surface->load_buffers();

} // end update dynamic mesh

void AppController::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
	Controller::key_callback(window,key,scancode,action,mods);
	if( action != GLFW_PRESS ){ return; }
	switch ( key ){
		case GLFW_KEY_SPACE: sim_running = !sim_running; break;
		case GLFW_KEY_S: save_single_ss = true; break;
		case GLFW_KEY_F: save_all_ss = !save_all_ss; break;
		case GLFW_KEY_P: sim_dostep = true; break;
		case GLFW_KEY_T: sim_realtime = !sim_realtime; break;
		case GLFW_KEY_H: print_help(); break;
	}

}

inline void AppController::print_help() const {
	std::stringstream ss;
	ss << "\n==========================================\nKeys:\n" <<
		"\t esc: exit app\n" <<
		"\t r: default camera position\n" <<
		"\t spacebar: toggle sim\n" <<
		"\t s: save a screenshot\n" <<
		"\t f: save a screenshot each frame\n" <<
		"\t p: run a single time step\n" <<
		"\t t: run sim in frame time\n" <<
	"==========================================\n";
	printf( "%s", ss.str().c_str() );
}


#endif
