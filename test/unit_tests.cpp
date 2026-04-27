// Copyright Matt Overby 2025.
// Distributed under the MIT License.

#include "ADMMPD.hpp"
#include <MCL/AssertHandler.hpp>
#include <MCL/Sort.hpp>
#include <MCL/XForm.hpp>
#include <iostream>
#include <stdlib.h>
#include <string>

template<typename T>
admmpd::ADMMPDData<T>
make_two_tets()
{
    admmpd::ADMMPDData<T> data;
    data.x.resize(5, 3);
    data.x <<         //
        -0.5,         //
        -0.5, 0,      //
        0.5, -0.5, 0, //
        0, 0.5, 0,    //
        0, 0, 1,      //
        0, 0, -1;     //
    data.tets.resize(2);
    data.tets[0].inds = { 0, 1, 2, 3 };
    data.tets[1].inds = { 0, 1, 4, 2 };
    data.masses = Eigen::VectorX<T>::Ones(5);
    return data;
}

int
main(int, char**)
{
    {
        // Rotation in zero gravity: no velocity
        admmpd::ADMMPDData<double> data = make_two_tets<double>();
        data.gravity = 0;
        admmpd::ADMMPDSolver<double>::initialize(data);
        mcl::XForm<double>::make_rotate(0.45, Eigen::Vector3d::Random().normalized()).apply(data.x);
        auto x0 = data.x;
        for (int i = 0; i < 10; ++i) {
            admmpd::ADMMPDSolver<double>::solve(data);
        }
        mclAssert((data.x - x0).lpNorm<Eigen::Infinity>() < 1e-7);
        mclAssert(data.v.lpNorm<Eigen::Infinity>() < 1e-7);
    }
    {
        // Free fall velocity with gravity
        // Note: choice of explicit prediction will change distance
        admmpd::ADMMPDData<double> data = make_two_tets<double>();
        data.timestep_seconds = 1.0 / 24.0;
        data.gravity = -9.8;
        admmpd::ADMMPDSolver<double>::initialize(data);
        for (int i = 0; i < 24; ++i) { // one second of simulation
            admmpd::ADMMPDSolver<double>::solve(data);
        }
        mclAssert(std::abs(data.v.row(0).norm() + data.gravity) < 1e-7);
    }

    return EXIT_SUCCESS;
}