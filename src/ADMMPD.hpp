// Copyright Matt Overby 2025.
// Distributed under the MIT License.

#ifndef ADMMPD_SOLVER_HPP
#define ADMMPD_SOLVER_HPP 1

#include "Energies.hpp"

#include <Eigen/SparseCholesky>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace admmpd {

/// @brief Wrapper for ADMM-PD constant linear system
template<typename T>
class LinearSystem
{
  public:
    LinearSystem();
    virtual ~LinearSystem() = default;
    LinearSystem& operator=(LinearSystem const& ls); ///< re-factorizes L
    LinearSystem(const LinearSystem<T>& ls);         ///< re-factorizes L

    using LDLT = Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>>;
    RowSparseMatrix<T> L;       ///<  mass-weighted constant Laplacian
    std::unique_ptr<LDLT> ldlt; ///<  factorization
};

/// @brief A hard constraint (row of constraint Jacobian C)
template<typename T, int S>
class LinearConstraint
{
  public:
    bool active;
    std::array<int, S> stencil;  ///< indices of constraint, negative if dof < S
    std::array<T, S * 3> coeffs; ///< coeffs for row of C
    T d;                         ///< rhs of Cx = d
    LinearConstraint()
    {
        active = false;
        stencil.fill(-1);
        coeffs.fill(T(0));
        d = T(0);
    }
};

/// @brief Container of all simulation data
/// Prior to ADMMPDSolver::initialize, must set the following data:
/// - x (rest shape and initial configuration)
/// - masses (per-vertex)
/// - springs and/or faces and/or tets (structural primitives)
/// Most settings (e.g., time step, ADMM weights) cannot be changed after
/// initialize, as they are baked into the constant Hessian.
template<typename T>
class ADMMPDData
{
  public:
    ADMMPDData() = default;
    virtual ~ADMMPDData() = default;

    /// @brief Optional convergence test, called every iteration
    std::function<bool(int)> converged;

    /// @brief Project vertex. If defined, solver defaults to MCGS.
    std::function<void(Eigen::Vector3<T>&)> project_vertex;

    /// @brief Detect collisions and generate linear constraints, called every iteration.
    std::function<void(const RowMatrixX3<T>&, std::vector<LinearConstraint<T, 4>>&)> collisions;

    T timestep_seconds = T(1.0 / 24.0); ///<  timestep in seconds, baked into L
    T gravity = T(-9.81);               ///<  y-acceleration of gravity m/s^2
    int max_iterations = 20;            ///<  max solver iterations per timestep
    int mcgs_iterations = 100;          ///<  max Gauss-Seidel iterations (if used)
    int max_threads = -1;               ///<  max thread count, -1 = auto
    int iter = 0;                       ///<  current solver iteration

    RowMatrixX3<T> x;         ///<  positions
    RowMatrixX3<T> x_rest;    ///<  initial configuration set on initialize
    RowMatrixX3<T> x_start;   ///<  positions at start of timestep
    ColMatrixX3<T> x_cm;      ///<  column major x (buffer for linear solve)
    RowMatrixX3<T> x_tilde;   ///<  explicit predictor
    RowMatrixX3<T> v;         ///<  velocity
    RowMatrixX3<T> z;         ///<  ADMM auxilary
    RowMatrixX3<T> z_prev;    ///<  last iteration ADMM auxilary
    RowMatrixX3<T> u;         ///<  ADMM dual
    RowMatrixX3<T> rhs;       ///<  linear solve rhs
    ColMatrixX3<T> rhs_cm;    ///<  linear solve rhs (buffer for linear solve)
    Eigen::VectorX<T> masses; ///<  per-vertex masses (kg)
    Eigen::SparseMatrix<T> D; ///<  reduction matrix

    std::vector<std::vector<int>> colors; ///<  vertex graph colors
    std::vector<bool> color_exec;         ///<  colors parallel execution

    std::vector<PinEnergy<T>> pins;                  ///< pin data
    std::vector<SpringEnergy<T>> springs;            ///< spring data
    std::vector<TriangleEnergy<T>> triangles;        ///< triangle data
    std::vector<TetEnergy<T>> tets;                  ///< tetrahedra data
    std::vector<BendEnergy<T>> hinges;               ///< hinge data
    std::vector<LinearConstraint<T, 4>> constraints; ///< hard constraints

    LinearSystem<T> linear_system; ///<  global-step linear system
    Eigen::VectorX<T> W_diag;      ///<  diagonal of weight matrix
    RowSparseMatrix<T> C;          ///< linear constraint matrix for Cx=d
    Eigen::VectorX<T> d;           ///< linear constraint rhs for Cx=d
    Eigen::VectorX<T> y;           ///< lagrange mults of Cx=d
};

/// @brief ADMM-PD solver functions
template<typename T>
class ADMMPDSolver
{
  public:
    /// @brief Computes Hessian, initializes energies and buffers
    static void initialize(ADMMPDData<T>& data);

    /// @brief Solves the time step (calls local_step, global_step), updates data.x
    static void solve(ADMMPDData<T>& data);

    /// @brief Computes z and u
    static void local_step(ADMMPDData<T>& data);

    /// @brief Computes x
    static void global_step(ADMMPDData<T>& data);

    /// @brief Compute primal (r) and dual residuals (s)
    static void residual(const ADMMPDData<T>& data, RowMatrixX3<T>& r, RowMatrixX3<T>& s);

    /// @brief Computes collision detection against self (tet meshes only) and floor
    static void update_constraints(ADMMPDData<T>& data);

    /// @brief Computes the constraint Jacobian of linearized constraints
    static void constraint_jacobian(ADMMPDData<T>& data);
};

} // end namespace admmpd

#ifndef ADMMPD_STATIC_LIBRARY
#include "ADMMPD.cpp"
#endif

#endif // ADMMPD_SOLVER_HPP
