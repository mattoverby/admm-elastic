// Copyright Matt Overby 2025.
// Distributed under the MIT License.

#include "ADMMPD.hpp"

#include <igl/opengl/glfw/Viewer.h>

#include <MCL/AssertHandler.hpp>
#include <MCL/BendingModel.hpp>
#include <MCL/ComputeMasses.hpp>
#include <MCL/FacesFromTets.hpp>
#include <MCL/ShapeFactory.hpp>
#include <MCL/VisualDebug.hpp>
#include <MCL/XForm.hpp>

#include <iostream>

void
init_stretch_energies(admmpd::ADMMPDData<double>& data, const Eigen::MatrixXi& F, bool as_springs = false)
{
    if (as_springs) {
        Eigen::MatrixXi E;
        mcl::get_unique_edges(F, E);
        mclAssert(E.rows() > 0);
        std::cout << "Num springs: " << E.rows() << std::endl;
        data.springs.resize(E.rows());
        for (int i = 0; i < int(E.rows()); ++i) {
            data.springs[i].inds = E.row(i).head<2>();
            data.springs[i].stiffness = 500;
        }
    } else {
        std::cout << "Num triangles: " << F.rows() << std::endl;
        data.triangles.resize(F.rows());
        for (int i = 0; i < int(F.rows()); ++i) {
            data.triangles[i].inds = F.row(i);
        }
    }
}

void
init_bend_energies(admmpd::ADMMPDData<double>& data, const Eigen::MatrixXi& F)
{
    admmpd::RowMatrixXi H;
    mcl::make_hinges(F, H);
    mclAssert(H.rows() > 0);
    std::cout << "Num hinges: " << H.rows() << std::endl;
    data.hinges.resize(H.rows());
    for (int i = 0; i < int(H.rows()); ++i) {
        data.hinges[i].inds = H.row(i);
        data.hinges[i].stiffness = 0.1;
    }
}

Eigen::Vector2i
get_pin_inds(Eigen::MatrixXd& V)
{
    // Lazily grap top corners to pin
    int top_left = -1;
    int top_right = -1;
    for (int i = 0; i < V.rows(); ++i) {
        if (V(i, 0) < -0.999 && V(i, 1) > 0.999) {
            top_left = i;
        } else if (V(i, 0) > 0.999 && V(i, 1) > 0.999) {
            top_right = i;
        }
    }
    mclAssert(top_left >= 0);
    mclAssert(top_right >= 0);
    return Eigen::Vector2i(top_left, top_right);
}

void
init_pins(admmpd::ADMMPDData<double>& data, Eigen::Vector2i& pins)
{
    data.pins.resize(2);
    data.pins[0].ind = pins[0];
    data.pins[0].position = data.x.row(pins[0]);
    data.pins[1].ind = pins[1];
    data.pins[1].position = data.x.row(pins[1]);
}

int
main(int, char**)
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    mcl::make_tri_quad(Eigen::Vector2d(-1, -1), Eigen::Vector2d(1, 1), 30, V, F);
    Eigen::Vector2i pin_inds = get_pin_inds(V);

    // Rotate so cloth is horizontal, reposition to drape on unit ball collider
    mcl::XForm<double> xf = mcl::XForm<double>::make_rotate(-90.0, Eigen::Vector3d(1, 0, 0));
    xf.apply(V);
    V.col(1).array() = 0.75;

    // Set up ADMM-PD data
    admmpd::ADMMPDData<double> data;
    data.x = V;
    init_pins(data, pin_inds); // call before rotation/translation
    mcl::compute_masses(V, F, data.masses);
    init_stretch_energies(data, F, false);
    init_bend_energies(data, F);

    // Sphere for collision.
    // Setting the project_vertex function defaults solver to MCGS
    Eigen::MatrixXd ballV;
    Eigen::MatrixXi ballF;
    Eigen::Vector3d ball_center(0, 0, 0);
    double ball_rad = 0.5;
    mcl::make_tri_sphere(ball_center, ball_rad, 2, ballV, ballF);
    data.project_vertex = [&](Eigen::Vector3d& xi) {
        const double eps = 0.01; // const offset to avoid visual artifacts
        Eigen::Vector3d delta = (xi - ball_center);
        if (delta.norm() < ball_rad + eps) {
            xi = ball_center + delta.normalized() * (ball_rad + eps);
        }
    };

    // Initialize
    admmpd::ADMMPDSolver<double>::initialize(data);

    // Use visual debugger to add points/etc for convenience
    mcl::VisualDebug& vd = mcl::VisualDebug::get_instance();
    if (data.pins.size() >= 2) {
        vd.add_point(data.x.row(data.pins[0].ind));
        vd.add_point(data.x.row(data.pins[1].ind));
    }

    // Viewer loop
    bool take_one_step = false;
    bool simulating = false;
    igl::opengl::glfw::Viewer viewer;
    viewer.core().is_animating = true;
    viewer.data().set_mesh(data.x, F);
    viewer.append_mesh();
    viewer.data(1).set_mesh(ballV, ballF);
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
        vd.set_data_igl(viewer, 2, false);
        return false;
    };

    std::cout << "\n\nPress:\n\t" << "\n\tspace to animate" << "\n\tS to step" << "\n\tR to reset" << "\n\n"
              << std::endl;
    viewer.launch();

    return EXIT_SUCCESS;
}