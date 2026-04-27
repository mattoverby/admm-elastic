// Copyright Matt Overby 2025.
// Distributed under the MIT License.

#include "ADMMPD.hpp"

#include <igl/opengl/glfw/Viewer.h>

#include <MCL/Centerize.hpp>
#include <MCL/ComputeMasses.hpp>
#include <MCL/EnergyModel.hpp>
#include <MCL/FacesFromTets.hpp>
#include <MCL/MicroTimer.hpp>
#include <MCL/ReadEleNode.hpp>

#include <iostream>

admmpd::ADMMPDData<double>
init_data(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T)
{
    admmpd::ADMMPDData<double> data;
    data.x = V;
    data.gravity = 0;
    mcl::compute_masses(V, T, data.masses);

    // Initialize with neo-Hookean
    data.tets.resize(T.rows());
    for (int i = 0; i < int(T.rows()); ++i) {
        data.tets[i].inds = T.row(i);
        data.tets[i].model = mcl::Lame<double>::soft_rubber(mcl::ENERGY_MODEL_NH);
    }

    return data;
}

void
pin_top_vertex(admmpd::ADMMPDData<double>& data)
{
    int top_vertex_index = -1;
    data.x.col(1).maxCoeff(&top_vertex_index);
    Eigen::Vector3d x0 = data.x.row(top_vertex_index);
    data.pins.resize(1);
    data.pins[0].ind = top_vertex_index;
    data.pins[0].position = x0;
    data.pins[0].stiffness = mcl::Lame<double>::rubber().bulk_modulus();
}

int
main(int, char**)
{
    // Load mesh
    std::string fn = ADMMPD_ROOT_DIR "/test/data/bunny_2250";
    Eigen::MatrixXd V;
    Eigen::MatrixXi T, F;
    if (!mcl::read_ele_node(fn, V, T)) {
        std::cout << "Failed to load " << fn << std::endl;
        return EXIT_FAILURE;
    }

    // Reshape
    mcl::centerize(V);
    mcl::scale_to_sphere(V, 1.0);
    mcl::faces_from_tets(T, F);

    // Set ADMMPD data
    admmpd::ADMMPDData<double> data = init_data(V, T);

    // Initialize solver
    mcl::MicroTimer timer;
    admmpd::ADMMPDSolver<double>::initialize(data);
    double elapsed_init = timer.elapsed_ms();
    std::cout << "Init solver: " << elapsed_init << " ms" << std::endl;

    // Default: randomized initialization, unit cube
    data.x = admmpd::RowMatrixX3<double>::Random(data.x.rows(), 3) * 0.5;

    // Init viewer
    bool take_one_step = false;
    bool simulating = false;
    igl::opengl::glfw::Viewer viewer;
    viewer.core().is_animating = true;
    viewer.data().set_mesh(data.x, F);
    viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer&, unsigned int key, int) -> bool {
        if (char(key) == ' ') {
            simulating = !simulating;
        } else if (char(key) == 's') {
            take_one_step = true;
            return true;
        } else if (char(key) == 'r') {
            data = init_data(V, T);
            admmpd::ADMMPDSolver<double>::initialize(data);
            data.x = admmpd::RowMatrixX3<double>::Random(data.x.rows(), 3) * 0.5;
        } else if (char(key) == 'p') {
            data = init_data(V, T);
            admmpd::ADMMPDSolver<double>::initialize(data);
            data.x.setZero();
        } else if (char(key) == 'h') {
            data = init_data(V, T);
            pin_top_vertex(data);
            data.gravity = -9.81;
            admmpd::ADMMPDSolver<double>::initialize(data);
        }
        return false;
    };
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer&) -> bool {
        if (simulating || take_one_step) {
            take_one_step = false;
            timer.reset();
            admmpd::ADMMPDSolver<double>::solve(data);
            double elapsed_solve = timer.elapsed_ms();
            std::cout << "Solve timestep: " << elapsed_solve << " ms" << std::endl;
        }
        viewer.data().clear();
        viewer.data().set_mesh(data.x, F);
        return false;
    };

    std::cout << "\n\nPress:\n\t" << "\n\tspace to animate" << "\n\tS to step" << "\n\tR to reset (random)"
              << "\n\tP to reset (point)" << "\n\tH to reset (hang from point)" << "\n\n"
              << std::endl;
    viewer.launch();

    return EXIT_SUCCESS;
}