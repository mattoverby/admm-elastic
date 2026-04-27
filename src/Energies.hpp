// Copyright Matt Overby 2025.
// Distributed under the MIT License.

#ifndef ADMMPD_ENERGIES_HPP
#define ADMMPD_ENERGIES_HPP 1

#include "MatrixTypes.hpp"

#include <MCL/Lame.hpp>
#include <vector>

namespace admmpd {

/// @brief Pin energies
template<typename T>
class PinEnergy
{
  public:
    int ind = -1;               ///< vertex index
    Eigen::Vector3<T> position; ///< vertex pin position
    T stiffness = 1000;         ///< vertex pin stiffness
    T weight = -1;              ///< ADMM weight, set internally
    int D_row;                  ///< starting row into D, set in admm_init

    /// @brief Initialize weight and global matrix coeffs; called in parallel.
    void admm_init(int D_row,
                   int D_trip,
                   const RowMatrixX3<T>& x,
                   Eigen::VectorX<T>& W_diag_sqrt,
                   std::vector<Eigen::Triplet<T>>& D_triplets);

    /// @brief Local step
    void admm_project(const RowMatrixX3<T>& x, RowMatrixX3<T>& z, RowMatrixX3<T>& u);
};

/// @brief Spring energies
template<typename T>
class SpringEnergy
{
  public:
    Eigen::Vector2i inds = { -1, -1 }; ///< spring indices
    T rest = -1;                       ///< spring rest length, if negative set internally
    T stiffness = 100;                 ///< spring stiffness
    T weight = -1;                     ///< ADMM weight, set internally
    int D_row;                         ///< starting row into D, set in admm_init

    /// @brief Initialize rest shape, weight, and global matrix coeffs; called in parallel.
    void admm_init(int D_row,
                   int D_trip,
                   const RowMatrixX3<T>& x,
                   Eigen::VectorX<T>& W_diag_sqrt,
                   std::vector<Eigen::Triplet<T>>& D_triplets);

    /// @brief Local step
    void admm_project(const RowMatrixX3<T>& x, RowMatrixX3<T>& z, RowMatrixX3<T>& u);
};

/// @brief Tet energies
template<typename T>
class TriangleEnergy
{
  public:
    Eigen::Vector3i inds = { -1, -1, -1 };                      ///< triangle indices
    T area = -1;                                                ///< triangle area
    T stiffness = 100;                                          ///< stretch stiffness
    T weight = -1;                                              ///< ADMM weight, set internally
    int D_row;                                                  ///< starting row into D, set in admm_init
    Eigen::Vector2<T> strain_limit = Eigen::Vector2<T>(1, -1);  ///< min and max strain limit
    Eigen::Matrix2<T> rest = Eigen::Matrix2<T>::Zero();         ///< rest shape
    Eigen::Matrix<T, 2, 3> Di = Eigen::Matrix<T, 2, 3>::Zero(); ///< local reduction matrix

    /// @brief Initialize rest shape, weight, and global matrix coeffs; called in parallel.
    void admm_init(int D_row,
                   int D_trip,
                   const RowMatrixX3<T>& x,
                   Eigen::VectorX<T>& W_diag_sqrt,
                   std::vector<Eigen::Triplet<T>>& D_triplets);

    /// @brief Local step, called in parallel.
    void admm_project(const RowMatrixX3<T>& x, RowMatrixX3<T>& z, RowMatrixX3<T>& u);
};

/// @brief Bend energies, currently only supported with Quadratic model.
template<typename T>
class BendEnergy
{
  public:
    Eigen::Vector4i inds = { -1, -1, -1, -1 }; ///< i0, i1 are shared-edge, i2, i3 are cross-edge
    T stiffness = T(0.01);                     ///< bend stiffness
};

/// @brief Tet energies
template<typename T>
class TetEnergy
{
  public:
    Eigen::Vector4i inds = { -1, -1, -1, -1 };             ///< tet indices
    T volume = -1;                                         ///< tet volume
    T weight = -1;                                         ///< ADMM weight, set in admm_init
    int D_row;                                             ///< starting row into D, set in admm_init
    Eigen::Matrix3<T> rest = Eigen::Matrix3<T>::Zero();    ///< rest shape
    mcl::Lame<T> model = mcl::Lame<T>::very_soft_rubber(); ///< strain model and stiffness params

    /// @brief Initialize rest shape, weight, and global matrix coeffs; called in parallel.
    void admm_init(int D_row,
                   int D_trip,
                   const RowMatrixX3<T>& x,
                   Eigen::VectorX<T>& W_diag_sqrt,
                   std::vector<Eigen::Triplet<T>>& D_triplets);

    /// @brief Local step, called in parallel.
    void admm_project(const RowMatrixX3<T>& x, RowMatrixX3<T>& z, RowMatrixX3<T>& u);

    /// @brief Minimizes local hyper-elastic proximal function
    void minimize_prox(const Eigen::Vector3<T>& S0, Eigen::Vector3<T>& S);
};

} // end namespace admmpd

#ifndef ADMMPD_STATIC_LIBRARY
#include "Energies.cpp"
#endif

#endif // ADMMPD_ENERGIES_HPP