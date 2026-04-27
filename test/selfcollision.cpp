// Copyright Matt Overby 2025.
// Distributed under the MIT License.

#include "Scene.hpp"

#include <igl/opengl/glfw/Viewer.h>

#include <MCL/Centerize.hpp>
#include <MCL/MicroTimer.hpp>
#include <MCL/ReadEleNode.hpp>
#include <MCL/ShapeFactory.hpp>

#include <iostream>

int
main(int, char**)
{

    admmpd::Scene<double> scene;
    scene.options.self_collision = true;
    int cube_refine = 2;

    // Load meshes
    {
        Eigen::MatrixXd V0, V1, V2, V3;
        Eigen::MatrixXi T0, T1, T2, T3;
        mcl::make_tet_box(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0.5, 0.5, 0.5), cube_refine, V0, T0);
        scene.add_tet_mesh(V0, T0);
        mcl::make_tet_box(Eigen::Vector3d(0.25, 0.75, 0.1), Eigen::Vector3d(0.75, 1.25, 0.6), cube_refine, V1, T1);
        scene.add_tet_mesh(V1, T1);
        if (mcl::read_ele_node(std::string(ADMMPD_ROOT_DIR "/test/data/bunny_2250"), V2, T2)) {
            mcl::centerize(V2);
            mcl::scale_to_sphere(V2, 0.75);
            V2.col(1).array() += 0.5;
            V2.col(2).array() += 0.5;
            scene.add_tet_mesh(V2, T2);
        }
        if (mcl::read_ele_node(std::string(ADMMPD_ROOT_DIR "/test/data/armadillo_3k"), V3, T3)) {
            mcl::centerize(V3);
            mcl::scale_to_sphere(V3, 1.0);
            V3.col(1).array() += 1.75;
            scene.add_tet_mesh(V3, T3);
        }
    }

    std::array<Eigen::RowVector3d, 4> mesh_colors = { Eigen::RowVector3d(1.00, 0.82, 0.86),
                                                      Eigen::RowVector3d(0.67, 0.94, 0.82),
                                                      Eigen::RowVector3d(0.65, 0.78, 0.91),
                                                      Eigen::RowVector3d(1.0, 1.0, 0.729) };

    // Keep everything in a sphere
    Eigen::MatrixXd Vsphere;
    Eigen::MatrixXi Fsphere;
    {
        Eigen::Vector3d center(0, 1, 0);
        double radius = 1.5;
        std::shared_ptr<admmpd::KinematicMesh<double>> sphere =
            std::make_shared<admmpd::KinematicSphere<double>>(radius, center, false);
        scene.add_kinematic_mesh(sphere);
        mcl::make_tri_sphere(center, radius, 2, Vsphere, Fsphere);
    }

    // Initialize solver
    scene.init_solver();

    // Viewer loop
    bool take_one_step = false;
    bool simulating = false;
    igl::opengl::glfw::Viewer viewer;
    viewer.core().is_animating = true;
    viewer.data().set_mesh(Vsphere, Fsphere);
    viewer.data().show_faces = false;
    for (int i = 0; i < scene.get_num_meshes(); ++i) {
        viewer.append_mesh(true);
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        scene.get_mesh(i, V, F);
        viewer.data(i + 1).set_mesh(V, F);
        viewer.data(i + 1).set_colors(mesh_colors[i % mesh_colors.size()]);
    }
    viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer&, unsigned int key, int) -> bool {
        if (char(key) == ' ') {
            simulating = !simulating;
        } else if (char(key) == 's') {
            take_one_step = true;
            return true;
        } else if (char(key) == 'r') {
            scene.init_solver();
        }
        return false;
    };
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer&) -> bool {
        if (simulating || take_one_step) {
            take_one_step = false;
            mcl::MicroTimer timer;
            scene.solve_timestep();
            double elapsed_step = timer.elapsed_ms();
            std::cout << "solve timestep: " << elapsed_step << " ms" << std::endl;
        }
        for (int i = 0; i < scene.get_num_meshes(); ++i) {
            Eigen::MatrixXd V;
            Eigen::MatrixXi F;
            scene.get_mesh(i, V, F);
            viewer.data(i + 1).clear();
            viewer.data(i + 1).set_mesh(V, F);
            viewer.data(i + 1).set_colors(mesh_colors[i % mesh_colors.size()]);
        }
        return false;
    };

    std::cout << "\n\nPress:\n\t" << "\n\tspace to animate" << "\n\tS to step" << "\n\tR to reset" << "\n\n"
              << std::endl;
    viewer.launch();

    return EXIT_SUCCESS;
}