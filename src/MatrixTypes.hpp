// Copyright Matt Overby 2025.
// Distributed under the MIT License.

#ifndef ADMMPD_MATRIXTYPES_HPP
#define ADMMPD_MATRIXTYPES_HPP 1

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace admmpd {

template<typename T>
using RowMatrixX3 = Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>;

template<typename T>
using RowMatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using RowMatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template<typename T>
using ColMatrixX3 = Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::ColMajor>;

template<typename T>
using RowSparseMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor>;

} // end namespace admmpd

#endif // ADMMPD_MATRIXTYPES_HPP