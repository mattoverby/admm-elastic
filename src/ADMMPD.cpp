// Copyright Matt Overby 2025.
// Distributed under the MIT License.

#include "ADMMPD.hpp"

#include <MCL/AssertHandler.hpp>
#include <MCL/BendingModel.hpp>
#include <MCL/ComputeMasses.hpp>
#include <MCL/GraphColor.hpp>
#include <MCL/KKTSolver.hpp>
#include <MCL/MultiColorGaussSeidel.hpp>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <iostream>

namespace admmpd {

template<typename T>
LinearSystem<T>::LinearSystem()
{
    ldlt = std::make_unique<LDLT>();
}

template<typename T>
LinearSystem<T>&
LinearSystem<T>::operator=(LinearSystem<T> const& ls)
{
    // LDLT factorizations are non-copyable, so we need to recompute.
    L = ls.L;
    ldlt = std::make_unique<LDLT>();
    if (ls.L.nonZeros() > 0) {
        ldlt->compute(L);
    }
    return *this;
}

template<typename T>
LinearSystem<T>::LinearSystem(const LinearSystem<T>& ls)
{
    *this = ls;
}

template<typename T>
void
ADMMPDSolver<T>::initialize(ADMMPDData<T>& data)
{
    // Counts
    int num_pins = int(data.pins.size());
    int num_springs = int(data.springs.size());
    int num_tris = int(data.triangles.size());
    int num_tets = int(data.tets.size());
    int num_verts = int(data.x.rows());
    int num_hinges = int(data.hinges.size());

    // Assert required input
    mclAssert(data.x.rows() > 0);
    mclAssert(data.masses.rows() == data.x.rows());
    mclAssert((data.masses.array() > 0).all());
    mclAssert((num_springs + num_tris + num_tets) > 0);
    mclAssert(data.project_vertex == nullptr || data.collisions == nullptr);

    // Initialize ADMM-PD data
    int D_rows = num_pins + num_springs + num_tris * 2 + num_tets * 3;
    int D_num_triplets = num_pins + num_springs * 2 + num_tris * 6 + num_tets * 12;
    std::vector<Eigen::Triplet<T>> D_triplets(D_num_triplets);
    data.W_diag = Eigen::VectorX<T>::Zero(D_rows);

    int D_row_pin_start = 0;
    int D_row_spring_start = D_row_pin_start + num_pins;
    int D_row_tri_start = D_row_spring_start + num_springs;
    int D_row_tet_start = D_row_tri_start + num_tris * 2;

    int D_trip_pin_start = 0;
    int D_trip_spring_start = D_trip_pin_start + num_pins;
    int D_trip_tri_start = D_trip_spring_start + num_springs * 2;
    int D_trip_tet_start = D_trip_tri_start + num_tris * 6;

    // For each energy, compute the rest state and reduction matrix.
    // Collect terms for the global matrix, dt^2*D'WWD
    tbb::task_arena arena(data.max_threads);
    arena.execute([&] {
        // pins: single triplet, single row
        tbb::parallel_for(tbb::blocked_range<int>(0, num_pins), [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i != range.end(); ++i) {
                int D_row = D_row_pin_start + i;
                int D_trip = D_trip_pin_start + i;
                auto& pin = data.pins[i];
                pin.admm_init(D_row, D_trip, data.x, data.W_diag, D_triplets);
            }
        });

        // springs, 2 triplets, single row
        tbb::parallel_for(tbb::blocked_range<int>(0, num_springs), [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i != range.end(); ++i) {
                int D_row = D_row_spring_start + i;
                int D_trip = D_trip_spring_start + i * 2;
                auto& spring = data.springs[i];
                spring.admm_init(D_row, D_trip, data.x, data.W_diag, D_triplets);
            }
        });

        // triangles: 6 triplets, 2 rows
        tbb::parallel_for(tbb::blocked_range<int>(0, num_tris), [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i != range.end(); ++i) {
                int D_row = D_row_tri_start + i * 2;
                int D_trip = D_trip_tri_start + i * 6;
                auto& tri = data.triangles[i];
                tri.admm_init(D_row, D_trip, data.x, data.W_diag, D_triplets);
            }
        });

        // tets: 12 triplets, 3 rows
        tbb::parallel_for(tbb::blocked_range<int>(0, num_tets), [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i != range.end(); ++i) {
                int D_row = D_row_tet_start + i * 3;
                int D_trip = D_trip_tet_start + i * 12;
                auto& tet = data.tets[i];
                tet.admm_init(D_row, D_trip, data.x, data.W_diag, D_triplets);
            }
        });
    }); // end arena

    // Compute reduction matrix (Dx = z)
    data.D.resize(D_rows, num_verts);
    data.D.setFromTriplets(D_triplets.begin(), D_triplets.end());
    data.D.makeCompressed();
    RowSparseMatrix<T> Dt = data.D.transpose();

    // Mass matrix, is there a better way to initialize this?
    Eigen::SparseMatrix<T> M(num_verts, num_verts);
    M.setIdentity();
    for (int i = 0; i < num_verts; ++i) {
        M.coeffRef(i, i) = data.masses[i];
    }

    // Bending Hessian is constant following Bergou et al.:
    // "A quadratic bending model for inextensible surfaces"
    Eigen::SparseMatrix<T> Q;
    Q.resize(num_verts, num_verts);
    if (num_hinges > 0) {
        std::vector<Eigen::Triplet<T>> Q_triplets;
        Q_triplets.reserve(num_hinges * 12);
        for (int i = 0; i < num_hinges; ++i) {
            auto hi = data.hinges[i].inds;
            std::array<Eigen::Vector3<T>, 4> xi = {
                data.x.row(hi[0]), data.x.row(hi[1]), data.x.row(hi[2]), data.x.row(hi[3])
            };
            Eigen::Matrix4<T> Qi = mcl::quadratic_bend_Q(xi[0], xi[1], xi[2], xi[3]);
            Qi *= data.hinges[i].stiffness;
            for (int j = 0; j < 4; ++j) {
                Q_triplets.emplace_back(hi[0], hi[j], Qi(0, j));
                Q_triplets.emplace_back(hi[1], hi[j], Qi(1, j));
                Q_triplets.emplace_back(hi[2], hi[j], Qi(2, j));
                Q_triplets.emplace_back(hi[3], hi[j], Qi(3, j));
            }
        }
        Q.setFromTriplets(Q_triplets.begin(), Q_triplets.end());
        Q.makeCompressed();
    }

    // ADMM-PD mass-weighted Laplacian
    T dt_dt = data.timestep_seconds * data.timestep_seconds;
    Eigen::SparseMatrix<T> A = dt_dt * Dt * data.W_diag.asDiagonal() * data.D;
    data.linear_system.L = M + A + Q;
    data.linear_system.L.makeCompressed();
    data.linear_system.ldlt->compute(data.linear_system.L);
    mclAssert(data.linear_system.ldlt->info() == Eigen::Success);

    // Graph coloring for parallel Gauss-Seidel
    // Combine colors below a size threshold to run in serial
    mcl::graph_color(A, data.colors);
    data.color_exec.resize(data.colors.size(), true);
    if (mcl::combine_small_colors(200, data.colors)) {
        data.color_exec.back() = false;
    }

    // Resize ADMM-PD data
    data.x_rest = data.x;
    data.x_start = data.x;
    data.x_cm = data.x;
    data.x_tilde = data.x;
    data.v = RowMatrixX3<T>::Zero(num_verts, 3);
    data.z = RowMatrixX3<T>::Zero(D_rows, 3);
    data.z_prev = RowMatrixX3<T>::Zero(D_rows, 3);
    data.u = RowMatrixX3<T>::Zero(D_rows, 3);
    data.rhs = RowMatrixX3<T>::Zero(num_verts, 3);
    data.rhs_cm = ColMatrixX3<T>::Zero(num_verts, 3);
}

template<typename T>
void
ADMMPDSolver<T>::solve(ADMMPDData<T>& data)
{
    // Compute explicit predictor
    data.x_start = data.x;
    if (std::abs(data.gravity) > 0) {
        data.v.col(1).array() += data.timestep_seconds * data.gravity;
    }
    data.x_tilde = data.x + data.timestep_seconds * data.v;
    data.x = data.x_tilde;

    // Solver loop
    data.iter = 0;
    while (data.iter < data.max_iterations) {
        local_step(data);
        update_constraints(data);
        global_step(data);
        data.iter++;
        if (data.converged != nullptr) {
            if (data.converged(data.iter)) {
                break;
            }
        }
    }

    // Update velocities
    data.v = (data.x - data.x_start) * (T(1) / data.timestep_seconds);
}

template<typename T>
void
ADMMPDSolver<T>::local_step(ADMMPDData<T>& data)
{
    // Computes z, u

    data.z_prev = data.z;
    int num_pins = int(data.pins.size());
    int num_springs = int(data.springs.size());
    int num_tris = int(data.triangles.size());
    int num_tets = int(data.tets.size());

    tbb::task_arena arena(data.max_threads);
    arena.execute([&] {
        // pins
        tbb::parallel_for(tbb::blocked_range<int>(0, num_pins), [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i != range.end(); ++i) {
                auto& pin = data.pins[i];
                pin.admm_project(data.x, data.z, data.u);
            }
        });

        // springs
        tbb::parallel_for(tbb::blocked_range<int>(0, num_springs), [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i != range.end(); ++i) {
                auto& spring = data.springs[i];
                spring.admm_project(data.x, data.z, data.u);
            }
        });

        // triangles
        tbb::parallel_for(tbb::blocked_range<int>(0, num_tris), [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i != range.end(); ++i) {
                auto& tri = data.triangles[i];
                tri.admm_project(data.x, data.z, data.u);
            }
        });

        // tets
        tbb::parallel_for(tbb::blocked_range<int>(0, num_tets), [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i != range.end(); ++i) {
                auto& tet = data.tets[i];
                tet.admm_project(data.x, data.z, data.u);
            }
        });
    }); // end arena
}

template<typename T>
void
ADMMPDSolver<T>::global_step(ADMMPDData<T>& data)
{
    // Computes x

    int num_verts = data.x.rows();
    T dt_dt = data.timestep_seconds * data.timestep_seconds;

    tbb::task_arena arena(data.max_threads);
    arena.execute([&] {
        // Compute Eq. 18: M x_tilde + dt^2 D'W'W(z-u)
        // Can be done with sparse matrix multiplies, but it's faster to handle all math in one parallel loop.
        tbb::parallel_for(tbb::blocked_range<int>(0, num_verts), [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                Eigen::RowVector3<T> sum = Eigen::RowVector3<T>::Zero();
                sum += data.masses[i] * data.x_tilde.row(i);

                // Multiply by cols of D (rows of Dt) to get dt^2 D'WW(z-u)
                for (typename Eigen::SparseMatrix<T>::InnerIterator it(data.D, i); it; ++it) {
                    int col = it.row();
                    T dtdt_DtWW_ij = dt_dt * it.value() * data.W_diag[col];
                    Eigen::RowVector3<T> zu = data.z.row(col) - data.u.row(col);
                    sum += dtdt_DtWW_ij * zu;
                }

                data.rhs.row(i) = sum;
            }
        });

        // If there are linear constraints, use CG from Sec. 5.1
        if (!data.constraints.empty()) {
            mcl::KKTSolver<Eigen::VectorX<T>, RowSparseMatrix<T>> kkt;

            // We'll reuse our factorization for the KKT solver.
            // However, we'll do the solve on row-major buffers to make the map back to vector easier.
            // Some annoying copies but that's not the bottleneck here. I think.
            RowMatrixX3<T> x_tmp = RowMatrixX3<T>::Zero(data.x.rows(), data.x.cols());
            RowMatrixX3<T> b_tmp = RowMatrixX3<T>::Zero(data.rhs.rows(), data.rhs.cols());
            kkt.solve_Axb = [&](const Eigen::VectorX<T>& b, Eigen::VectorX<T>& x) {
                b_tmp = Eigen::Map<RowMatrixX3<T>>((T*)(b.data()), b.size() / 3, 3);
                tbb::parallel_for(0, 3, [&](int i) {                             //
                    x_tmp.col(i) = data.linear_system.ldlt->solve(b_tmp.col(i)); //
                });
                x = Eigen::Map<Eigen::VectorX<T>>(x_tmp.data(), x_tmp.size());
            };

            // Map variables to vector for the KKT solve
            constraint_jacobian(data);
            Eigen::SparseMatrix<T> A; // not needed, LDLT is provided
            Eigen::VectorX<T> x = Eigen::Map<Eigen::VectorX<T>>(data.x.data(), data.x.size());
            Eigen::VectorX<T> b = Eigen::Map<Eigen::VectorX<T>>(data.rhs.data(), data.rhs.size());
            int kkt_iters = kkt.solve(A, b, data.C, data.d, x, data.y);
            mclAssert(kkt_iters > 0);
            data.x = Eigen::Map<RowMatrixX3<T>>(x.data(), data.x.rows(), data.x.cols());
        }
        // If there is a projection operator, use Gauss-Seidel from Sec. 5.3
        else if (data.project_vertex != nullptr) {
            mcl::MultiColorGaussSeidel<RowMatrixX3<T>> mcgs;
            mcgs.options.max_iters = data.mcgs_iterations;
            mcgs.project = [&](int ind, RowMatrixX3<T>& X) {
                Eigen::Vector3<T> xi = X.row(ind);
                data.project_vertex(xi);
                X.row(ind) = xi;
            };
            mcgs.solve(data.linear_system.L, data.rhs, data.x, data.colors, data.color_exec);
        }
        // Eq. 31: linear solve, buffered in column-major
        else {
            data.rhs_cm = data.rhs;
            tbb::parallel_for(0, 3, [&](int i) {                                       //
                data.x_cm.col(i) = data.linear_system.ldlt->solve(data.rhs_cm.col(i)); //
            });
            data.x = data.x_cm;
        }
    }); // arena
}

template<typename T>
void
ADMMPDSolver<T>::residual(const ADMMPDData<T>& data, RowMatrixX3<T>& r, RowMatrixX3<T>& s)
{
    // Eqs. 21 and 22, can be used for convergence test
    Eigen::VectorX<T> W_sqrt = data.W_diag.array().sqrt();
    r = W_sqrt.asDiagonal() * (data.D * data.x - data.z);
    s = data.D.transpose() * data.W_diag.asDiagonal() * (data.z - data.z_prev);
}

template<typename T>
void
ADMMPDSolver<T>::update_constraints(ADMMPDData<T>& data)
{
    data.constraints.clear();
    if (data.collisions != nullptr) {
        data.collisions(data.x, data.constraints);
    }
}

template<typename T>
void
ADMMPDSolver<T>::constraint_jacobian(ADMMPDData<T>& data)
{
    data.C.setZero();
    data.d.setZero();
    if (data.constraints.empty()) {
        return;
    }

    std::vector<Eigen::Triplet<T>> C_triplets;
    std::vector<T> d_coeffs;
    C_triplets.reserve(data.constraints.size() * 12);
    d_coeffs.reserve(data.constraints.size());

    // If we solve for Cx=d exactly, we'll get a lot of jittering. That's because
    // constriants are resolved on one iteration, and cease to exist in another.
    // Instead, we'll reduce the delta so the elements stay in slight contact, and
    // aren't fully resolved w/ Cx=d (unless they resolve naturally through momentum).
    for (const auto& c : data.constraints) {
        if (!c.active) {
            continue;
        }
        int C_row = d_coeffs.size();
        for (int i = 0; i < 4; ++i) {
            if (c.stencil[i] < 0) {
                continue;
            }
            C_triplets.emplace_back(C_row, c.stencil[i] * 3 + 0, c.coeffs[i * 3 + 0]);
            C_triplets.emplace_back(C_row, c.stencil[i] * 3 + 1, c.coeffs[i * 3 + 1]);
            C_triplets.emplace_back(C_row, c.stencil[i] * 3 + 2, c.coeffs[i * 3 + 2]);
        }
        d_coeffs.emplace_back(c.d);
    }

    data.C.resize(d_coeffs.size(), data.x.size()); // C is full dof
    data.C.setFromTriplets(C_triplets.begin(), C_triplets.end());
    data.d = Eigen::Map<Eigen::VectorX<T>>(d_coeffs.data(), d_coeffs.size());
}

} // end namespace admmpd

#ifdef ADMMPD_STATIC_LIBRARY
template class admmpd::LinearSystem<double>;
template class admmpd::LinearSystem<float>;
template class admmpd::ADMMPDData<double>;
template class admmpd::ADMMPDData<float>;
template class admmpd::ADMMPDSolver<double>;
template class admmpd::ADMMPDSolver<float>;
#endif
