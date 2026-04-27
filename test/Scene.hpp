#ifndef ADMMPD_SCENE_HPP
#define ADMMPD_SCENE_HPP

#include "ADMMPD.hpp"

#include <MCL/AssertHandler.hpp>
#include <MCL/BVHTree.hpp>
#include <MCL/Barycoords.hpp>
#include <MCL/ComputeMasses.hpp>
#include <MCL/FacesFromTets.hpp>
#include <MCL/NarrowPhase.hpp>
#include <MCL/Normal.hpp>
#include <MCL/SDF.hpp>
#include <MCL/SignedMeasure.hpp>

#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>

#include <vector>

namespace admmpd {

/// @brief Tetmesh with collision detection
template<typename T>
class TetMesh
{
  public:
    admmpd::RowMatrixX3<T> vertices; // rest
    admmpd::RowMatrixXi tets, faces;
    mcl::ccd::SDF<T> sdf; // SDF at rest
    TetMesh(const admmpd::RowMatrixX3<T>& vertices_, const admmpd::RowMatrixXi& tets_)
        : vertices(vertices_)
        , tets(tets_)
    {
        mcl::faces_from_tets(tets, faces);
        sdf.create(vertices, faces);
    }
};

template<typename T>
class KinematicMesh
{
  public:
    /// @brief Signed distance returns {signed_distance, gradient}
    virtual std::tuple<T, Eigen::Vector3<T>> signed_distance(const Eigen::Vector3<T>& point) = 0;
};

template<typename T>
class KinematicSphere : public KinematicMesh<T>
{
  public:
    T radius;
    Eigen::Vector3<T> center;
    bool project_out; ///< if vertex inside cylinder, project it out

    /// @brief Constructor
    KinematicSphere(T radius_, const Eigen::Vector3<T>& center_, bool project_out_ = true)
        : radius(radius_)
        , center(center_)
        , project_out(project_out_)
    {
    }

    /// @brief Signed distance returns {signed_distance, gradient}
    std::tuple<T, Eigen::Vector3<T>> signed_distance(const Eigen::Vector3<T>& point)
    {
        Eigen::Matrix<T, 3, 1> dir = point - center;
        T mag = dir.norm();
        if (mag < T(1e-7)) {
            return { T(0), Eigen::Matrix<T, 3, 1>::Zero() };
        }
        Eigen::Matrix<T, 3, 1> n = dir / mag;
        T sd = mag - radius; // signed distance (positive outside)
        if (!project_out) {
            sd = -sd;
            n = -n;
        }
        return { sd, n };
    }
};

/// @brief Traverser for point-in-tet BVH queries
template<typename T>
class VertexInTetMesh : public mcl::ccd::BVHTraverse<T, 3, 4>
{
  public:
    using VolumeType = typename mcl::ccd::BVHTraverse<T, 3, 4>::VolumeType;
    using ObjectType = typename mcl::ccd::BVHTraverse<T, 3, 4>::ObjectType;
    int vertex_index;
    const admmpd::RowMatrixX3<T>& vertices;
    const admmpd::RowMatrixXi& tets;
    int in_tet;
    Eigen::Vector4<T> in_barys;
    Eigen::Vector3<T> point;

    VertexInTetMesh(int vertex_index_, const admmpd::RowMatrixX3<T>& vertices_, const admmpd::RowMatrixXi& tets_)
        : vertex_index(vertex_index_)
        , vertices(vertices_)
        , tets(tets_)
        , in_tet(-1)
        , in_barys(-1, -1, -1, -1)
    {
        point = vertices.row(vertex_index);
    }

    ~VertexInTetMesh() = default;

    bool intersectVolume(const VolumeType& v) { return v.contains(point); }

    bool intersectObject(const ObjectType& o)
    {
        Eigen::RowVector4i tet = tets.row(o.idx);
        if (tet[0] == vertex_index || tet[1] == vertex_index || tet[2] == vertex_index || tet[3] == vertex_index) {
            return false; // skip self
        }
        auto barys = mcl::point_tet_barys<T>(
            point, vertices.row(tet[0]), vertices.row(tet[1]), vertices.row(tet[2]), vertices.row(tet[3]));
        if (barys.minCoeff() >= T(0) && barys.maxCoeff() <= T(1)) {
            in_tet = o.idx;
            in_barys = barys;
            return true; // stop traversing
        }
        return false;
    }
};

/// @brief Helper class for multi-mesh scenes and collision detection
template<typename T>
class Scene
{
  public:
    struct Options
    {
        T floor_y = std::numeric_limits<T>::lowest(); ///< y-value of the floor
        bool self_collision = false;                  ///< process self collisions
    } options;

    /// @brief Adds a deformable tet mesh to the scene
    void add_tet_mesh(const Eigen::MatrixX<T>& vertices, const Eigen::MatrixXi& tets)
    {
        meshes.emplace_back(vertices, tets); // copies to row major
    }

    /// @brief Adds a kinematic mesh to the scene
    void add_kinematic_mesh(std::shared_ptr<KinematicMesh<T>>& mesh) { kinematic_meshes.emplace_back(mesh); }

    /// @brief Returns ADMMPDData
    ADMMPDData<T>& get_admmpd_data() { return admmpd_data; }

    /// @brief Returns the number of meshes in the scene.
    int get_num_meshes() const { return meshes.size(); }

    /// @brief Returns deformed vertices and faces for the mesh.
    void get_mesh(int mesh_index, Eigen::MatrixXd& V, Eigen::MatrixXi& F) const
    {
        if (mesh_index >= get_num_meshes() || admmpd_data.x.rows() == 0) {
            return;
        }
        int v_offset = 0;
        for (int i = 0; i < mesh_index; ++i) {
            v_offset += meshes[i].vertices.rows();
        }
        V = admmpd_data.x.block(v_offset, 0, meshes[mesh_index].vertices.rows(), 3);
        F = meshes[mesh_index].faces;
    }

    /// @brief Takes one timestep
    void solve_timestep() { admmpd::ADMMPDSolver<T>::solve(admmpd_data); }

    /// @brief Initializes solver after all meshes have been added.
    void init_solver()
    {
        admmpd_data = admmpd::ADMMPDData<T>();
        int nv = 0;
        int nt = 0;
        int nf = 0;
        int nsv = 0;
        for (const auto& mesh : meshes) {
            nv += mesh.vertices.rows();
            nt += mesh.tets.rows();
            nf += mesh.faces.rows();
        }

        rest.resize(nv, 3);
        tets.resize(nt, 4);
        faces.resize(nf, 3);
        tet_to_mesh.resize(nt);
        faces_offset = Eigen::VectorXi::Zero(meshes.size() + 1);
        int start_vi = 0;
        int start_ti = 0;
        int start_fi = 0;
        int start_svi = 0;
        for (size_t mesh_index = 0; mesh_index < meshes.size(); ++mesh_index) {
            const auto& mesh = meshes[mesh_index];
            rest.block(start_vi, 0, mesh.vertices.rows(), 3) = mesh.vertices;
            tets.block(start_ti, 0, mesh.tets.rows(), 4) = mesh.tets;
            tets.block(start_ti, 0, mesh.tets.rows(), 4).array() += start_vi;
            faces.block(start_fi, 0, mesh.faces.rows(), 3) = mesh.faces;
            faces.block(start_fi, 0, mesh.faces.rows(), 3).array() += start_vi;
            tet_to_mesh.segment(start_ti, mesh.tets.rows()).array() = mesh_index;
            start_vi += mesh.vertices.rows();
            start_ti += mesh.tets.rows();
            start_fi += mesh.faces.rows();
            faces_offset[mesh_index + 1] = start_fi;
        }

        surface_vertices = mcl::get_unique_vertices(faces);

        // Init ADMM-PD data and options
        mcl::compute_masses(rest, tets, admmpd_data.masses);
        admmpd_data.x = rest;
        admmpd_data.timestep_seconds = 1.0 / 100.0;
        admmpd_data.max_iterations = 10;
        admmpd_data.tets.resize(tets.rows());
        for (int i = 0; i < int(tets.rows()); ++i) {
            admmpd_data.tets[i].inds = tets.row(i);
            admmpd_data.tets[i].model = mcl::Lame<T>::very_soft_rubber();
        }

        // Function to generate collision constraints
        admmpd_data.collisions = [&](const RowMatrixX3<T>& x, std::vector<LinearConstraint<T, 4>>& c) {
            this->detect_collisions(x, c);
        };

        admmpd::ADMMPDSolver<T>::initialize(admmpd_data);
    }

    void detect_collisions(const RowMatrixX3<T>& x, std::vector<LinearConstraint<T, 4>>& constraints)
    {
        // Floor:
        for (int i = 0; i < x.rows(); ++i) {
            if (x(i, 1) <= options.floor_y) {
                LinearConstraint<T, 4> c;
                c.active = true;
                c.stencil[0] = i;
                c.coeffs[1] = 1; // n=(0,1,0)
                c.d = options.floor_y;
                constraints.emplace_back(c);
            }
        }

        if (!options.self_collision && kinematic_meshes.empty()) {
            return;
        }

        if (options.self_collision) {
            tet_tree.update(x.data(), x.data(), tets.data(), tets.rows());
        }

        // Tuple: {vertex, tet, barys} if self collision, {vertex, -1, {normal, offset}} if kinematic
        tbb::concurrent_vector<std::tuple<int, int, Eigen::Vector4<T>>> vertex_in_mesh;
        vertex_in_mesh.reserve(surface_vertices.size());

        // Collision broad phase: find points in tet or inside kinematic object.
        // Only query the surface of the deformable tet meshes.
        tbb::parallel_for(
            tbb::blocked_range<int>(0, surface_vertices.size()), [&](const tbb::blocked_range<int>& range) {
                for (int i = range.begin(); i != range.end(); ++i) {
                    int surface_vertex_index = surface_vertices[i];
                    const Eigen::Vector3<T> point = x.row(surface_vertex_index);

                    // Kinematic mesh collision
                    for (auto& kinematic_mesh : kinematic_meshes) {
                        auto [signed_distance, sdf_grad] = kinematic_mesh->signed_distance(point);
                        if (signed_distance < T(0)) {
                            T d = -signed_distance + sdf_grad.dot(point);
                            Eigen::Vector4<T> coeffs(sdf_grad[0], sdf_grad[1], sdf_grad[2], d);
                            vertex_in_mesh.emplace_back(surface_vertex_index, -1, coeffs);
                        }
                    }

                    // Self collision
                    if (options.self_collision) {
                        VertexInTetMesh<T> traverser(surface_vertex_index, x, tets);
                        tet_tree.traverse(&traverser);
                        if (traverser.in_tet >= 0) {
                            vertex_in_mesh.emplace_back(surface_vertex_index, traverser.in_tet, traverser.in_barys);
                        }
                    }
                }
            });

        // Collision narrow phase: create a constraint for every point-in-mesh.
        // If we encounter an error with the SDF, deactive the constraint.
        int constraint_offset = constraints.size();
        constraints.resize(constraints.size() + vertex_in_mesh.size());
        tbb::parallel_for(tbb::blocked_range<int>(0, vertex_in_mesh.size()), [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i != range.end(); ++i) {
                const auto& [vertex_index, tet_index, coeffs] = vertex_in_mesh[i];
                auto& c = constraints[constraint_offset + i];

                // Kinematic mesh collision
                if (tet_index < 0) {
                    c.active = true;
                    c.stencil[0] = vertex_index;
                    c.coeffs[0] = coeffs[0]; // n0
                    c.coeffs[1] = coeffs[1]; // n1
                    c.coeffs[2] = coeffs[2]; // n2
                    c.d = coeffs[3];
                    continue;
                }

                // Self collision
                auto tet = tets.row(tet_index);
                int mesh_index = tet_to_mesh[tet_index];

                // Map deformed vertices to rest shape
                std::array<Eigen::Vector3<T>, 4> qx_rest = {
                    rest.row(tet[0]), rest.row(tet[1]), rest.row(tet[2]), rest.row(tet[3])
                };
                Eigen::Vector3<T> px_rest =
                    coeffs[0] * qx_rest[0] + coeffs[1] * qx_rest[1] + coeffs[2] * qx_rest[2] + coeffs[3] * qx_rest[3];

                // Each SDF will return indices local to that mesh
                auto [local_face_index, tri_barys, dist] = meshes[mesh_index].sdf.project_to_surface(px_rest);
                if (local_face_index < 0) {
                    continue; // error with SDF
                }

                // Find normal of nearest triangle
                int global_face_index = faces_offset[mesh_index] + local_face_index;
                auto tri = faces.row(global_face_index);
                std::array<Eigen::Vector3<T>, 3> qx_face = { x.row(tri[0]), x.row(tri[1]), x.row(tri[2]) };
                Eigen::Vector3<T> normal = mcl::triangle_normal(qx_face[0], qx_face[1], qx_face[2]);

                // If we are already on the positive side of the normal, no need to form the constraint.
                // We are not solving this as a QP with inequality constraints. It treats the linearized constraint
                // as an equality constraint, so not filtering contacts this way will get sticking artifacts.
                Eigen::Vector3<T> px = x.row(vertex_index);
                if (normal.dot(px - qx_face[0]) > 0) {
                    continue;
                }

                // Linearize the constraint
                c.active = true;
                c.stencil[0] = vertex_index;
                c.stencil[1] = tri[0];
                c.stencil[2] = tri[1];
                c.stencil[3] = tri[2];
                c.d = 0;
                for (int j = 0; j < 3; ++j) {
                    c.coeffs[j] = normal[j];
                    c.coeffs[3 + j] = -tri_barys[0] * normal[j];
                    c.coeffs[6 + j] = -tri_barys[1] * normal[j];
                    c.coeffs[9 + j] = -tri_barys[2] * normal[j];
                }
            }
        });
    }

  protected:
    admmpd::ADMMPDData<T> admmpd_data;                               ///< solver data
    std::vector<TetMesh<T>> meshes;                                  ///< wrapper for SDF
    std::vector<std::shared_ptr<KinematicMesh<T>>> kinematic_meshes; ///< collision meshes
    mcl::ccd::BVHTree<T, 3, 4> tet_tree;                             ///< all meshes in one tree
    admmpd::RowMatrixX3<T> rest;                                     //< initial configuration
    admmpd::RowMatrixXi tets, faces;                                 //< all tets and faces in scene
    Eigen::VectorXi surface_vertices;                                //< all surface vertices in scene
    Eigen::VectorXi tet_to_mesh;                                     //< tet index to mesh index
    Eigen::VectorXi faces_offset;                                    //< per-mesh + 1 offset of face indices
    Eigen::VectorX<T> masses;                                        //< all per-vertex masses
};

} // endif admmpd

#endif