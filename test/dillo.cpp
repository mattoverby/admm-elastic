// Copyright Matt Overby 2025.
// Distributed under the MIT License.

#include "ADMMPD.hpp"

#include <igl/opengl/glfw/Viewer.h>

#include <MCL/Centerize.hpp>
#include <MCL/ComputeMasses.hpp>
#include <MCL/FacesFromTets.hpp>
#include <MCL/ReadEleNode.hpp>
#include <MCL/ShapeFactory.hpp>

#include <iostream>

void
init_tets(const Eigen::MatrixXi& T, admmpd::ADMMPDData<double>& data)
{
    data.tets.resize(T.rows());
    for (int i = 0; i < int(T.rows()); ++i) {
        data.tets[i].inds = T.row(i);
        data.tets[i].model = mcl::Lame<double>(1000000, 0.299);
    }
}

int
main(int, char**)
{
    // Load mesh
    std::string fn = ADMMPD_ROOT_DIR "/test/data/armadillo_3k";
    Eigen::MatrixXd V;
    Eigen::MatrixXi T, F;
    if (!mcl::read_ele_node(fn, V, T)) {
        std::cout << "Failed to load " << fn << std::endl;
        return EXIT_FAILURE;
    }

    mcl::centerize(V);
    mcl::scale_to_sphere(V, 0.5);
    mcl::faces_from_tets(T, F);

    // Init ADMM-PD data and options
    admmpd::ADMMPDData<double> data;
    data.timestep_seconds = 1.0 / 100.0;
    data.max_iterations = 10;
    data.mcgs_iterations = 50;
    data.x = V;
    mcl::compute_masses(V, T, data.masses);
    init_tets(T, data);

    // Setting a project_vertex operator will trigger global step to use
    // multi-color Gauss-Seidel
    double floor_y = -1;
    data.project_vertex = [&](Eigen::Vector3d& xi) { xi[1] = std::max(xi[1], floor_y); };

    // Initialize solver
    admmpd::ADMMPDSolver<double>::initialize(data);

    // Create mesh for floor rendering
    Eigen::MatrixXd floorV;
    Eigen::MatrixXi floorF;
    Eigen::Vector3d floor_bmin(-1, -1.1, -1);
    Eigen::Vector3d floor_bmax(1, floor_y, 1);
    mcl::make_tri_box(floor_bmin, floor_bmax, floorV, floorF);

    // Viewer loop
    bool take_one_step = false;
    bool simulating = false;
    igl::opengl::glfw::Viewer viewer;
    viewer.core().is_animating = true;
    viewer.data().set_mesh(data.x, F);
    viewer.append_mesh();
    viewer.data(1).set_mesh(floorV, floorF);
    viewer.data(1).set_colors(Eigen::RowVector3d(0, 1, 0));
    viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer&, unsigned int key, int) -> bool {
        if (char(key) == ' ') {
            simulating = !simulating;
        } else if (char(key) == 's') {
            take_one_step = true;
            return true;
        } else if (char(key) == 'r') {
            data.x = V;
            admmpd::ADMMPDSolver<double>::initialize(data);
        }
        return false;
    };
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer&) -> bool {
        if (simulating || take_one_step) {
            take_one_step = false;
            admmpd::ADMMPDSolver<double>::solve(data);
        }
        viewer.data(0).clear();
        viewer.data(0).set_mesh(data.x, F);
        return false;
    };

    std::cout << "\n\nPress:\n\t" << "\n\tspace to animate" << "\n\tS to step" << "\n\tR to reset" << "\n\n"
              << std::endl;
    viewer.launch();

    return EXIT_SUCCESS;
}