
#include "Energies.hpp"

#include <MCL/AssertHandler.hpp>
#include <MCL/EnergyModel.hpp>
#include <MCL/SignedSVD.hpp>

#include <sstream>

namespace admmpd {

template<typename T>
void
PinEnergy<T>::admm_init(int D_row_,
                        int D_trip,
                        const RowMatrixX3<T>& x,
                        Eigen::VectorX<T>& W_diag,
                        std::vector<Eigen::Triplet<T>>& D_triplets)
{
    mclAssert(ind >= 0);
    mclAssert(ind < x.rows());
    D_row = D_row_;
    D_triplets[D_trip] = Eigen::Triplet<T>(D_row, ind, T(1));
    weight = stiffness;
    W_diag[D_row] = weight;
}

template<typename T>
void
PinEnergy<T>::admm_project(const RowMatrixX3<T>& x, RowMatrixX3<T>& z, RowMatrixX3<T>& u)
{
    Eigen::Vector3<T> xi = x.row(ind);
    Eigen::Vector3<T> ui = u.row(D_row);
    Eigen::Vector3<T> zi = position;
    ui += (xi - zi);
    z.row(D_row) = zi;
    u.row(D_row) = ui;
}

template<typename T>
void
SpringEnergy<T>::admm_init(int D_row_,
                           int D_trip,
                           const RowMatrixX3<T>& x,
                           Eigen::VectorX<T>& W_diag,
                           std::vector<Eigen::Triplet<T>>& D_triplets)
{
    mclAssert(inds.minCoeff() >= 0);
    mclAssert(inds.maxCoeff() < x.rows());
    D_row = D_row_;
    D_triplets[D_trip] = Eigen::Triplet<T>(D_row, inds[0], T(1));
    D_triplets[D_trip + 1] = Eigen::Triplet<T>(D_row, inds[1], T(-1));
    if (rest < 0) {
        rest = (x.row(inds[0]) - x.row(inds[1])).norm();
    }
    weight = stiffness * rest;
    W_diag[D_row] = weight;
}

template<typename T>
void
SpringEnergy<T>::admm_project(const RowMatrixX3<T>& x, RowMatrixX3<T>& z, RowMatrixX3<T>& u)
{

    Eigen::Vector3<T> x0 = x.row(inds[0]);
    Eigen::Vector3<T> x1 = x.row(inds[1]);
    Eigen::Vector3<T> Di_x = x0 - x1;
    Eigen::Vector3<T> ui = u.row(D_row);
    Eigen::Vector3<T> zi = Di_x + ui;

    // Project
    Eigen::Vector3<T> p = rest * zi.normalized();
    T k_v = stiffness * rest;
    zi = (weight * p + weight * zi) / (k_v + weight);

    // Update
    ui += (Di_x - zi);
    u.row(D_row) = ui;
    z.row(D_row) = zi;
}

template<typename T>
void
TriangleEnergy<T>::admm_init(int D_row_,
                             int D_trip,
                             const RowMatrixX3<T>& x,
                             Eigen::VectorX<T>& W_diag,
                             std::vector<Eigen::Triplet<T>>& D_triplets)
{
    mclAssert(inds.minCoeff() >= 0);
    mclAssert(inds.maxCoeff() < x.rows());
    D_row = D_row_;
    std::array<Eigen::Vector3<T>, 3> xi;
    xi[0] = x.row(inds[0]);
    xi[1] = x.row(inds[1]);
    xi[2] = x.row(inds[2]);

    // Rest shape
    if (area <= T(0) || rest.squaredNorm() <= T(0)) {
        Eigen::Vector3<T> e12 = xi[1] - xi[0];
        Eigen::Vector3<T> e13 = xi[2] - xi[0];
        Eigen::Vector3<T> n1 = e12.normalized();
        Eigen::Vector3<T> n2 = (e13 - e13.dot(n1) * n1).normalized();
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 3, 2> edges;
        basis.col(0) = n1;
        basis.col(1) = n2;
        edges.col(0) = e12;
        edges.col(1) = e13;
        rest = (basis.transpose() * edges).inverse();
        area = T(0.5) * ((xi[1] - xi[0]).cross(xi[2] - xi[0])).norm();
    }

    mclAssert(area > 0);

    // Selector
    Eigen::Matrix<T, 3, 2> S = Eigen::Matrix<T, 3, 2>::Zero();
    S.row(0).array() = -1;
    S(1, 0) = 1;
    S(2, 1) = 1;

    // Reduction matrix
    weight = stiffness * area;
    Di = (S * rest).transpose();
    for (int r = 0; r < 2; ++r) {
        int D_row_r = D_row + r;
        int trip_ind = D_trip + r * 3;
        W_diag[D_row_r] = weight;
        D_triplets[trip_ind + 0] = Eigen::Triplet<T>(D_row_r, inds[0], Di(r, 0));
        D_triplets[trip_ind + 1] = Eigen::Triplet<T>(D_row_r, inds[1], Di(r, 1));
        D_triplets[trip_ind + 2] = Eigen::Triplet<T>(D_row_r, inds[2], Di(r, 2));
    }
}

template<typename T>
void
TriangleEnergy<T>::admm_project(const RowMatrixX3<T>& x, RowMatrixX3<T>& z, RowMatrixX3<T>& u)
{
    Eigen::Matrix<T, 3, 3> xi;
    xi.row(0) = x.row(inds[0]);
    xi.row(1) = x.row(inds[1]);
    xi.row(2) = x.row(inds[2]);
    Eigen::Matrix<T, 2, 3> Di_x = Di * xi;
    Eigen::Matrix<T, 2, 3> ui = u.template block<2, 3>(D_row, 0);
    Eigen::Matrix<T, 2, 3> zi = (Di_x + ui);

    T k_v = stiffness * area;
    Eigen::JacobiSVD<Eigen::Matrix<T, 3, 2>> svd((zi.transpose()).eval(), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<T, 2, 3> P = (svd.matrixU().template leftCols<2>() * svd.matrixV().transpose()).transpose();
    zi = (k_v * P + weight * zi) / (k_v + weight);

#if 0 // TODO
    if (strain_limit[0] < strain_limit[1]) {
        T l_col0 = zi.row(0).norm();
        T l_col1 = zi.row(1).norm();
        if (l_col0 < strain_limit[0]) {
            zi.row(0) *= (strain_limit[0] / std::max(l_col0, T(1e-8)));
        }
        if (l_col1 < strain_limit[0]) {
            zi.row(1) *= (strain_limit[0] / std::max(l_col1, T(1e-8)));
        }
        if (l_col0 > strain_limit[1]) {
            zi.row(0) *= (strain_limit[1] / std::max(l_col0, T(1e-8)));
        }
        if (l_col1 > strain_limit[1]) {
            zi.row(1) *= (strain_limit[1] / std::max(l_col1, T(1e-8)));
        }
    }
#endif
    // Apply
    ui += (Di_x - zi);
    u.row(D_row + 0) = ui.row(0);
    u.row(D_row + 1) = ui.row(1);
    z.row(D_row + 0) = zi.row(0);
    z.row(D_row + 1) = zi.row(1);
}

template<typename T>
void
TetEnergy<T>::admm_init(int D_row_,
                        int D_trip,
                        const RowMatrixX3<T>& x,
                        Eigen::VectorX<T>& W_diag,
                        std::vector<Eigen::Triplet<T>>& D_triplets)
{
    mclAssert(inds.minCoeff() >= 0);
    mclAssert(inds.maxCoeff() < x.rows());
    D_row = D_row_;
    std::array<Eigen::Vector3<T>, 4> xi;
    xi[0] = x.row(inds[0]);
    xi[1] = x.row(inds[1]);
    xi[2] = x.row(inds[2]);
    xi[3] = x.row(inds[3]);

    // Rest shape and volume
    if (volume <= T(0) || rest.squaredNorm() <= T(0)) {
        Eigen::Matrix3<T> edges;
        edges.col(0) = xi[1] - xi[0];
        edges.col(1) = xi[2] - xi[0];
        edges.col(2) = xi[3] - xi[0];
        rest = edges.inverse();
        volume = edges.determinant() / 6.0;
    }

    mclAssert(volume > 0);

    // Selector
    Eigen::Matrix<T, 4, 3> S = Eigen::Matrix<T, 4, 3>::Zero();
    S.row(0).array() = -1;
    S.template block<3, 3>(1, 0).diagonal().array() = 1;

    // Reduction matrix
    weight = model.bulk_modulus() * volume;
    Eigen::Matrix<T, 3, 4> D = (S * rest).transpose();
    for (int r = 0; r < 3; ++r) {
        int D_row_r = D_row + r;
        int trip_ind = D_trip + r * 4;
        W_diag[D_row_r] = weight;
        D_triplets[trip_ind + 0] = Eigen::Triplet<T>(D_row_r, inds[0], D(r, 0));
        D_triplets[trip_ind + 1] = Eigen::Triplet<T>(D_row_r, inds[1], D(r, 1));
        D_triplets[trip_ind + 2] = Eigen::Triplet<T>(D_row_r, inds[2], D(r, 2));
        D_triplets[trip_ind + 3] = Eigen::Triplet<T>(D_row_r, inds[3], D(r, 3));
    }
}

template<typename T>
void
TetEnergy<T>::admm_project(const RowMatrixX3<T>& x, RowMatrixX3<T>& z, RowMatrixX3<T>& u)
{
    Eigen::RowVector3<T> x0 = x.row(inds[0]);
    Eigen::Matrix3<T> edges;
    edges.col(0) = x.row(inds[1]) - x0;
    edges.col(1) = x.row(inds[2]) - x0;
    edges.col(2) = x.row(inds[3]) - x0;
    Eigen::Matrix3<T> ui = u.template block<3, 3>(D_row, 0);
    Eigen::Matrix3<T> Di_x = (edges * rest).transpose();
    Eigen::Matrix3<T> zi = Di_x + ui;

    // SVD
    Eigen::Vector3<T> S;
    Eigen::Matrix3<T> U, V;
    mcl::signed_svd<T, 3>(zi, S, U, V);
    Eigen::Vector3<T> S0 = S;
    S = Eigen::Vector3<T>::Ones();

    // Solve for S
    if (model.model() == mcl::ENERGY_MODEL_ARAP) {
        T k_v = model.bulk_modulus() * volume;
        Eigen::Matrix3<T> P = U * S.asDiagonal() * V.transpose();
        zi = (k_v * P + weight * zi) / (k_v + weight);
    } else {
        minimize_prox(S0, S);
        zi = U * S.asDiagonal() * V.transpose();
    }

    // Update
    ui += (Di_x - zi);
    u.row(D_row + 0) = ui.row(0);
    u.row(D_row + 1) = ui.row(1);
    u.row(D_row + 2) = ui.row(2);
    z.row(D_row + 0) = zi.row(0);
    z.row(D_row + 1) = zi.row(1);
    z.row(D_row + 2) = zi.row(2);
}

template<typename T>
void
TetEnergy<T>::minimize_prox(const Eigen::Vector3<T>& S0, Eigen::Vector3<T>& S)
{
    // Use Newton's method to minimize the proximal function.
    // The matrix tends to be ill-conditioned, so it's worth considering
    // alternatives if runtime is a concern. In the original paper we used L-BFGS,
    // which seems silly for a 3x3 linear solve, but it worked just fine.
    const T grad_tol = 1e-6;
    const T rel_delta_tol = 1e-6;
    const int max_iters = 10; // probably hits max iters every time
    T admm_stiffness = model.bulk_modulus();
    mcl::EnergyModel<3, T> energy_model(model);
    Eigen::Vector3<T> grad, p, S_prev;
    Eigen::Matrix3<T> H;

    int iter = 0;
    for (; iter < max_iters; ++iter) {
        // Gradient
        grad.setZero();
        T value_prev = energy_model.gradient(S, grad);
        value_prev += (admm_stiffness * 0.5) * (S - S0).squaredNorm();
        grad += admm_stiffness * (S - S0);

        // Descent with Newton's method.
        // Matrix can be ill-conditioned so using QR.
        H.setZero();
        energy_model.hessian(S, H);
        H.diagonal().array() += admm_stiffness;
        p = H.householderQr().solve(-grad);

        // Initial step
        T alpha = T(1);
        S_prev = S;
        S = S_prev + alpha * p;
        T value_k = energy_model.energy_density(S);
        value_k += (admm_stiffness * 0.5) * (S - S0).squaredNorm();

        // Lazy line search, break with any decrease in energy
        int ls_iter = 0;
        while (value_k > value_prev) {
            alpha *= T(0.5);
            S = S_prev + alpha * p;
            value_k = energy_model.energy_density(S);
            value_k += (admm_stiffness * 0.5) * (S - S0).squaredNorm();
            ls_iter++;
        }

        if (grad.norm() < grad_tol) {
            break;
        }

        if (std::abs((value_prev - value_k) / value_prev) < rel_delta_tol) {
            break;
        }
    }
}

} // end namespace admmpd

#ifdef ADMMPD_STATIC_LIBRARY
template class admmpd::PinEnergy<double>;
template class admmpd::PinEnergy<float>;
template class admmpd::SpringEnergy<double>;
template class admmpd::SpringEnergy<float>;
template class admmpd::TriangleEnergy<double>;
template class admmpd::TriangleEnergy<float>;
template class admmpd::TetEnergy<double>;
template class admmpd::TetEnergy<float>;
#endif