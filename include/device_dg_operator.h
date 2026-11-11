// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) XXXX - YYYY by the polyDEAL authors
//
// This file is part of the polyDEAL library.
//
// Detailed license information governing the source code
// can be found in LICENSE.md at the top level directory.
//
// -----------------------------------------------------------------------------

#ifndef polydeal_device_dg_operator_h
#define polydeal_device_dg_operator_h

#include <deal.II/base/config.h>

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <Kokkos_Core.hpp>

#include <map>
#include <memory>
#include <vector>

namespace dealii
{
  /**
   * Matrix-free symmetric interior penalty (SIP) Laplace operator for FE_DGQ
   * on a standard quad/hex mesh, with all per-application work done in Kokkos
   * kernels in the MemorySpace::Default memory space.
   *
   * The discrete form is the one of
   * Utils::MatrixFreeOperators::LaplaceOperatorDG (deal.II step-59 kernels),
   * including its penalty definition, so that the two operators agree to
   * rounding and this class can replace the assembled fine matrix in the
   * agglomerated multigrid pipeline.
   *
   * Design. The kernel is cell-centric: one work item per locally owned cell
   * evaluates the cell integral and all its face integrals and writes only the
   * rows of its own degrees of freedom. Interior faces are therefore visited
   * from both sides, which doubles the face work but removes any need for
   * atomics, colouring or a compress() of the destination. Faces between a
   * locally owned and a ghost cell are handled by the owning rank of each
   * side, using ghost values of the source vector.
   *
   * Shape functions are not stored. The tensor-product Lagrange basis of
   * FE_DGQ is evaluated on the device at reference points that are stored per
   * quadrature point, on the cell as well as on both sides of every face.
   * This costs more arithmetic than sum factorisation, but it makes no
   * assumption on the position of quadrature points inside the reference
   * cell. The identical kernel structure is what a polytopal operator needs,
   * where quadrature points are pulled back into a bounding box.
   *
   * Hanging nodes are not supported.
   */
  template <int dim, int degree, typename Number = double>
  class DeviceDGOperator : public EnableObserverPointer
  {
  public:
    using value_type  = Number;
    using size_type   = types::global_dof_index;
    using KokkosSpace = typename MemorySpace::Default::kokkos_space;
    using ExecSpace   = typename KokkosSpace::execution_space;
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
    using PartitionerPtr = std::shared_ptr<const Utilities::MPI::Partitioner>;

    static constexpr unsigned int n_1d    = degree + 1;
    static constexpr unsigned int n_dofs  = Utilities::pow(degree + 1, dim);
    static constexpr unsigned int n_faces = 2 * dim;

    DeviceDGOperator() = default;

    /**
     * Set up the operator for @p dof_handler, which must use FE_DGQ<dim>
     * of the given @p degree. Cell integrals use QGauss<dim>(degree + 1) and
     * face integrals QGauss<dim - 1>(degree + 1), as the host operator does.
     */
    void
    reinit(const Mapping<dim> &mapping, const DoFHandler<dim> &dof_handler);

    void
    clear();

    size_type
    m() const;

    size_type
    n() const;

    MPI_Comm
    get_mpi_communicator() const;

    const PartitionerPtr &
    get_domain_partitioner() const;

    /**
     * Initialize @p v with locally owned DoFs plus the ghost DoFs of the
     * neighbouring cells, the layout the operator expects as source.
     */
    void
    initialize_dof_vector(VectorType &v) const;

    void
    vmult(VectorType &dst, const VectorType &src) const;

    void
    vmult_add(VectorType &dst, const VectorType &src) const;

    /**
     * The operator is symmetric.
     */
    void
    Tvmult(VectorType &dst, const VectorType &src) const;

    void
    Tvmult_add(VectorType &dst, const VectorType &src) const;

    void
    get_diagonal(VectorType &diagonal) const;

    void
    get_inverse_diagonal(VectorType &inverse_diagonal) const;

    Number
    el(const size_type, const size_type) const;

    unsigned int
    n_locally_owned_cells() const;

    std::size_t
    memory_consumption() const;

  private:
    template <typename T>
    using View1 = Kokkos::View<T *, KokkosSpace>;
    template <typename T>
    using View2 = Kokkos::View<T **, Kokkos::LayoutLeft, KokkosSpace>;
    template <typename T>
    using View3 = Kokkos::View<T ***, Kokkos::LayoutLeft, KokkosSpace>;
    template <typename T>
    using View4 = Kokkos::View<T ****, Kokkos::LayoutLeft, KokkosSpace>;
    template <typename T>
    using View5 = Kokkos::View<T *****, Kokkos::LayoutLeft, KokkosSpace>;

    /**
     * All device data, bundled so that it can be captured by value in
     * kernels.
     */
    struct DeviceData
    {
      // 1D Lagrange nodes of the element
      View1<Number> nodes_1d;

      // (cell, dof): local index into a vector with the domain layout, for
      // locally owned cells followed by the ghost cells that neighbour them
      View2<unsigned int> dof_indices;

      // cell quadrature, locally owned cells only
      View3<Number> cell_q_unit; // (cell, q, d)
      View2<Number> cell_q_JxW;  // (cell, q)
      View4<Number> cell_q_invJ; // (cell, q, d, e) = d xi_d / d x_e

      // faces of locally owned cells
      View2<unsigned int> face_neighbor; // (cell, f): row of dof_indices, or
                                         // invalid_unsigned_int at boundary
      View2<Number> face_sigma;          // (cell, f)
      View4<Number> face_q_unit_m;       // (cell, f, q, d) this side
      View4<Number> face_q_unit_p;       // (cell, f, q, d) neighbour side
      View3<Number> face_q_JxW;          // (cell, f, q)
      View4<Number> face_q_normal;       // (cell, f, q, d) outward of cell
      View5<Number> face_q_invJ_m;       // (cell, f, q, d, e)
      View5<Number> face_q_invJ_p;       // (cell, f, q, d, e)

      unsigned int n_owned_cells = 0;
      unsigned int n_cell_q      = 0;
      unsigned int n_face_q      = 0;
    };

    /**
     * Evaluate the 1D Lagrange basis and its derivative at @p x.
     */
    KOKKOS_INLINE_FUNCTION static void
    lagrange_1d(const View1<Number> &nodes,
                const Number         x,
                Number              *val,
                Number              *der);

    /**
     * Evaluate all shape functions and their reference gradients at the
     * reference point (x[0], ..., x[dim-1]).
     */
    KOKKOS_INLINE_FUNCTION static void
    shape_values_and_gradients(const View1<Number> &nodes,
                               const Number        *x,
                               Number              *phi,
                               Number              *dphi);

    void
    apply(VectorType &dst, const VectorType &src, const bool add) const;

    DeviceData     data;
    PartitionerPtr partitioner;
    MPI_Comm       communicator  = MPI_COMM_SELF;
    size_type      n_global_dofs = 0;

    mutable VectorType tmp_src;
  };



  // ------------------------------ implementation ---------------------------

  template <int dim, int degree, typename Number>
  KOKKOS_INLINE_FUNCTION void
  DeviceDGOperator<dim, degree, Number>::lagrange_1d(const View1<Number> &nodes,
                                                     const Number         x,
                                                     Number              *val,
                                                     Number              *der)
  {
    for (unsigned int k = 0; k < n_1d; ++k)
      {
        val[k] = 1.;
        der[k] = 0.;
        for (unsigned int j = 0; j < n_1d; ++j)
          if (j != k)
            {
              const Number inv = Number(1.) / (nodes(k) - nodes(j));
              const Number f   = (x - nodes(j)) * inv;
              der[k]           = der[k] * f + val[k] * inv;
              val[k] *= f;
            }
      }
  }



  template <int dim, int degree, typename Number>
  KOKKOS_INLINE_FUNCTION void
  DeviceDGOperator<dim, degree, Number>::shape_values_and_gradients(
    const View1<Number> &nodes,
    const Number        *x,
    Number              *phi,
    Number              *dphi)
  {
    Number val[dim][n_1d];
    Number der[dim][n_1d];
    for (unsigned int d = 0; d < dim; ++d)
      lagrange_1d(nodes, x[d], val[d], der[d]);

    for (unsigned int i = 0; i < n_dofs; ++i)
      {
        unsigned int idx[dim];
        unsigned int rest = i;
        for (unsigned int d = 0; d < dim; ++d)
          {
            idx[d] = rest % n_1d;
            rest /= n_1d;
          }

        Number v = 1.;
        for (unsigned int d = 0; d < dim; ++d)
          v *= val[d][idx[d]];
        phi[i] = v;

        for (unsigned int d = 0; d < dim; ++d)
          {
            Number g = der[d][idx[d]];
            for (unsigned int e = 0; e < dim; ++e)
              if (e != d)
                g *= val[e][idx[e]];
            dphi[i * dim + d] = g;
          }
      }
  }



  template <int dim, int degree, typename Number>
  void
  DeviceDGOperator<dim, degree, Number>::clear()
  {
    data          = DeviceData();
    partitioner   = nullptr;
    communicator  = MPI_COMM_SELF;
    n_global_dofs = 0;
    tmp_src.reinit(0);
  }



  template <int dim, int degree, typename Number>
  void
  DeviceDGOperator<dim, degree, Number>::reinit(
    const Mapping<dim>    &mapping,
    const DoFHandler<dim> &dof_handler)
  {
    clear();

    const FiniteElement<dim> &fe = dof_handler.get_fe();
    AssertThrow(dynamic_cast<const FE_DGQ<dim> *>(&fe) != nullptr,
                ExcMessage("DeviceDGOperator requires FE_DGQ."));
    AssertThrow(fe.degree == static_cast<unsigned int>(degree),
                ExcMessage("The polynomial degree of the DoFHandler does not "
                           "match the template parameter."));
    AssertDimension(fe.n_dofs_per_cell(), n_dofs);

    const Triangulation<dim> &tria = dof_handler.get_triangulation();
    communicator                   = tria.get_mpi_communicator();
    n_global_dofs                  = dof_handler.n_dofs();

    // ----- parallel layout -----
    const IndexSet &owned = dof_handler.locally_owned_dofs();
    const IndexSet  relevant =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    partitioner = std::make_shared<Utilities::MPI::Partitioner>(owned,
                                                                relevant,
                                                                communicator);

    // ----- enumerate locally owned cells and their ghost neighbours -----
    std::vector<unsigned int> cell_row(tria.n_active_cells(),
                                       numbers::invalid_unsigned_int);
    std::vector<typename DoFHandler<dim>::active_cell_iterator> rows;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_row[cell->active_cell_index()] = rows.size();
          rows.push_back(cell);
        }
    const unsigned int n_owned = rows.size();
    for (unsigned int c = 0; c < n_owned; ++c)
      for (const unsigned int f : rows[c]->face_indices())
        if (!rows[c]->at_boundary(f))
          {
            const auto neighbor = rows[c]->neighbor(f);
            AssertThrow(neighbor->is_active() &&
                          !rows[c]->neighbor_is_coarser(f),
                        ExcMessage("Hanging nodes are not supported."));
            if (cell_row[neighbor->active_cell_index()] ==
                numbers::invalid_unsigned_int)
              {
                cell_row[neighbor->active_cell_index()] = rows.size();
                rows.push_back(neighbor);
              }
          }
    const unsigned int n_rows = rows.size();

    const QGauss<dim>     cell_quadrature(degree + 1);
    const QGauss<dim - 1> face_quadrature(degree + 1);
    const unsigned int    n_cell_q = cell_quadrature.size();
    const unsigned int    n_face_q = face_quadrature.size();

    data.n_owned_cells = n_owned;
    data.n_cell_q      = n_cell_q;
    data.n_face_q      = n_face_q;

    // ----- allocate device views and host mirrors -----
    data.nodes_1d = View1<Number>("polydeal::dg::nodes_1d", n_1d);
    data.dof_indices =
      View2<unsigned int>("polydeal::dg::dof_indices", n_rows, n_dofs);
    data.cell_q_unit =
      View3<Number>("polydeal::dg::cell_q_unit", n_owned, n_cell_q, dim);
    data.cell_q_JxW =
      View2<Number>("polydeal::dg::cell_q_JxW", n_owned, n_cell_q);
    data.cell_q_invJ =
      View4<Number>("polydeal::dg::cell_q_invJ", n_owned, n_cell_q, dim, dim);
    data.face_neighbor =
      View2<unsigned int>("polydeal::dg::face_neighbor", n_owned, n_faces);
    data.face_sigma =
      View2<Number>("polydeal::dg::face_sigma", n_owned, n_faces);
    data.face_q_unit_m = View4<Number>(
      "polydeal::dg::face_q_unit_m", n_owned, n_faces, n_face_q, dim);
    data.face_q_unit_p = View4<Number>(
      "polydeal::dg::face_q_unit_p", n_owned, n_faces, n_face_q, dim);
    data.face_q_JxW =
      View3<Number>("polydeal::dg::face_q_JxW", n_owned, n_faces, n_face_q);
    data.face_q_normal = View4<Number>(
      "polydeal::dg::face_q_normal", n_owned, n_faces, n_face_q, dim);
    data.face_q_invJ_m = View5<Number>(
      "polydeal::dg::face_q_invJ_m", n_owned, n_faces, n_face_q, dim, dim);
    data.face_q_invJ_p = View5<Number>(
      "polydeal::dg::face_q_invJ_p", n_owned, n_faces, n_face_q, dim, dim);

    auto h_nodes         = Kokkos::create_mirror_view(data.nodes_1d);
    auto h_dof_indices   = Kokkos::create_mirror_view(data.dof_indices);
    auto h_cell_q_unit   = Kokkos::create_mirror_view(data.cell_q_unit);
    auto h_cell_q_JxW    = Kokkos::create_mirror_view(data.cell_q_JxW);
    auto h_cell_q_invJ   = Kokkos::create_mirror_view(data.cell_q_invJ);
    auto h_face_neighbor = Kokkos::create_mirror_view(data.face_neighbor);
    auto h_face_sigma    = Kokkos::create_mirror_view(data.face_sigma);
    auto h_face_q_unit_m = Kokkos::create_mirror_view(data.face_q_unit_m);
    auto h_face_q_unit_p = Kokkos::create_mirror_view(data.face_q_unit_p);
    auto h_face_q_JxW    = Kokkos::create_mirror_view(data.face_q_JxW);
    auto h_face_q_normal = Kokkos::create_mirror_view(data.face_q_normal);
    auto h_face_q_invJ_m = Kokkos::create_mirror_view(data.face_q_invJ_m);
    auto h_face_q_invJ_p = Kokkos::create_mirror_view(data.face_q_invJ_p);

    // 1D nodes: FE_DGQ numbers DoFs lexicographically with x fastest, so the
    // first n_1d support points carry the 1D nodes in their x coordinate.
    {
      const std::vector<Point<dim>> &support = fe.get_unit_support_points();
      AssertDimension(support.size(), n_dofs);
      for (unsigned int k = 0; k < n_1d; ++k)
        h_nodes(k) = support[k][0];
    }

    // DoF indices of owned and neighbouring ghost cells
    {
      std::vector<types::global_dof_index> dofs(n_dofs);
      for (unsigned int c = 0; c < n_rows; ++c)
        {
          rows[c]->get_dof_indices(dofs);
          for (unsigned int i = 0; i < n_dofs; ++i)
            h_dof_indices(c, i) = partitioner->global_to_local(dofs[i]);
        }
    }

    // Cell data
    {
      FEValues<dim> fe_values(mapping,
                              fe,
                              cell_quadrature,
                              update_JxW_values | update_inverse_jacobians);
      for (unsigned int c = 0; c < n_owned; ++c)
        {
          fe_values.reinit(rows[c]);
          for (unsigned int q = 0; q < n_cell_q; ++q)
            {
              for (unsigned int d = 0; d < dim; ++d)
                h_cell_q_unit(c, q, d) = cell_quadrature.point(q)[d];
              h_cell_q_JxW(c, q) = fe_values.JxW(q);
              const DerivativeForm<1, dim, dim> &invJ =
                fe_values.inverse_jacobian(q);
              for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int e = 0; e < dim; ++e)
                  h_cell_q_invJ(c, q, d, e) = invJ[d][e];
            }
        }
    }

    // Face data
    {
      const Number penalty_factor =
        Number(std::max(degree, 1) * (degree + 1.0));

      FEFaceValues<dim> fe_face(mapping,
                                fe,
                                face_quadrature,
                                update_JxW_values | update_normal_vectors |
                                  update_quadrature_points |
                                  update_inverse_jacobians);

      // |n . grad xi_k| at a point, with k the reference direction normal to
      // the face: the quantity deal.II's matrix-free penalty is built from.
      const auto normal_scaling = [](const Tensor<1, dim>              &n,
                                     const DerivativeForm<1, dim, dim> &invJ,
                                     const unsigned int face_no) {
        const unsigned int k = face_no / 2;
        Number             s = 0;
        for (unsigned int j = 0; j < dim; ++j)
          s += n[j] * invJ[k][j];
        return std::abs(s);
      };

      std::vector<Point<dim>> unit_points_p(n_face_q);

      for (unsigned int c = 0; c < n_owned; ++c)
        for (const unsigned int f : rows[c]->face_indices())
          {
            fe_face.reinit(rows[c], f);
            const std::vector<Point<dim>> &real_points =
              fe_face.get_quadrature_points();

            for (unsigned int q = 0; q < n_face_q; ++q)
              {
                const Point<dim> unit_m =
                  mapping.transform_real_to_unit_cell(rows[c], real_points[q]);
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    h_face_q_unit_m(c, f, q, d) = unit_m[d];
                    h_face_q_normal(c, f, q, d) = fe_face.normal_vector(q)[d];
                  }
                h_face_q_JxW(c, f, q) = fe_face.JxW(q);
                const DerivativeForm<1, dim, dim> &invJ =
                  fe_face.inverse_jacobian(q);
                for (unsigned int d = 0; d < dim; ++d)
                  for (unsigned int e = 0; e < dim; ++e)
                    h_face_q_invJ_m(c, f, q, d, e) = invJ[d][e];
              }

            const Number pen_m = normal_scaling(fe_face.normal_vector(0),
                                                fe_face.inverse_jacobian(0),
                                                f);

            if (rows[c]->at_boundary(f))
              {
                h_face_neighbor(c, f) = numbers::invalid_unsigned_int;
                h_face_sigma(c, f)    = pen_m * penalty_factor * Number(2.);
                for (unsigned int q = 0; q < n_face_q; ++q)
                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      h_face_q_unit_p(c, f, q, d) = 0.;
                      for (unsigned int e = 0; e < dim; ++e)
                        h_face_q_invJ_p(c, f, q, d, e) = 0.;
                    }
                continue;
              }

            const auto         neighbor = rows[c]->neighbor(f);
            const unsigned int nf       = rows[c]->neighbor_of_neighbor(f);
            h_face_neighbor(c, f) = cell_row[neighbor->active_cell_index()];

            for (unsigned int q = 0; q < n_face_q; ++q)
              {
                unit_points_p[q] =
                  mapping.transform_real_to_unit_cell(neighbor, real_points[q]);
                for (unsigned int d = 0; d < dim; ++d)
                  h_face_q_unit_p(c, f, q, d) = unit_points_p[q][d];
              }

            FEValues<dim> fe_neighbor(mapping,
                                      fe,
                                      Quadrature<dim>(unit_points_p),
                                      update_inverse_jacobians);
            fe_neighbor.reinit(neighbor);
            for (unsigned int q = 0; q < n_face_q; ++q)
              {
                const DerivativeForm<1, dim, dim> &invJ =
                  fe_neighbor.inverse_jacobian(q);
                for (unsigned int d = 0; d < dim; ++d)
                  for (unsigned int e = 0; e < dim; ++e)
                    h_face_q_invJ_p(c, f, q, d, e) = invJ[d][e];
              }

            const Number pen_p = normal_scaling(fe_face.normal_vector(0),
                                                fe_neighbor.inverse_jacobian(0),
                                                nf);
            h_face_sigma(c, f) = (pen_m + pen_p) * penalty_factor;
          }
    }

    Kokkos::deep_copy(data.nodes_1d, h_nodes);
    Kokkos::deep_copy(data.dof_indices, h_dof_indices);
    Kokkos::deep_copy(data.cell_q_unit, h_cell_q_unit);
    Kokkos::deep_copy(data.cell_q_JxW, h_cell_q_JxW);
    Kokkos::deep_copy(data.cell_q_invJ, h_cell_q_invJ);
    Kokkos::deep_copy(data.face_neighbor, h_face_neighbor);
    Kokkos::deep_copy(data.face_sigma, h_face_sigma);
    Kokkos::deep_copy(data.face_q_unit_m, h_face_q_unit_m);
    Kokkos::deep_copy(data.face_q_unit_p, h_face_q_unit_p);
    Kokkos::deep_copy(data.face_q_JxW, h_face_q_JxW);
    Kokkos::deep_copy(data.face_q_normal, h_face_q_normal);
    Kokkos::deep_copy(data.face_q_invJ_m, h_face_q_invJ_m);
    Kokkos::deep_copy(data.face_q_invJ_p, h_face_q_invJ_p);
    Kokkos::fence();
  }



  template <int dim, int degree, typename Number>
  inline typename DeviceDGOperator<dim, degree, Number>::size_type
  DeviceDGOperator<dim, degree, Number>::m() const
  {
    return n_global_dofs;
  }



  template <int dim, int degree, typename Number>
  inline typename DeviceDGOperator<dim, degree, Number>::size_type
  DeviceDGOperator<dim, degree, Number>::n() const
  {
    return n_global_dofs;
  }



  template <int dim, int degree, typename Number>
  inline MPI_Comm
  DeviceDGOperator<dim, degree, Number>::get_mpi_communicator() const
  {
    return communicator;
  }



  template <int dim, int degree, typename Number>
  inline const typename DeviceDGOperator<dim, degree, Number>::PartitionerPtr &
  DeviceDGOperator<dim, degree, Number>::get_domain_partitioner() const
  {
    return partitioner;
  }



  template <int dim, int degree, typename Number>
  inline unsigned int
  DeviceDGOperator<dim, degree, Number>::n_locally_owned_cells() const
  {
    return data.n_owned_cells;
  }



  template <int dim, int degree, typename Number>
  void
  DeviceDGOperator<dim, degree, Number>::initialize_dof_vector(
    VectorType &v) const
  {
    Assert(partitioner, ExcNotInitialized());
    v.reinit(partitioner);
  }



  template <int dim, int degree, typename Number>
  void
  DeviceDGOperator<dim, degree, Number>::apply(VectorType       &dst,
                                               const VectorType &src,
                                               const bool        add) const
  {
    Assert(partitioner, ExcNotInitialized());
    AssertDimension(dst.locally_owned_size(),
                    partitioner->locally_owned_size());
    AssertDimension(src.locally_owned_size(),
                    partitioner->locally_owned_size());

    const VectorType *src_ptr = &src;
    if (src.get_partitioner().get() != partitioner.get() &&
        !src.get_partitioner()->is_compatible(*partitioner))
      {
        if (tmp_src.get_partitioner().get() != partitioner.get())
          tmp_src.reinit(partitioner);
        tmp_src.copy_locally_owned_data_from(src);
        src_ptr = &tmp_src;
      }

    const bool had_ghosts = src_ptr->has_ghost_elements();
    if (!had_ghosts)
      src_ptr->update_ghost_values();

    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    const Kokkos::View<const Number *, KokkosSpace, Unmanaged> src_view(
      src_ptr->get_values(),
      partitioner->locally_owned_size() + partitioner->n_ghost_indices());
    Kokkos::View<Number *, KokkosSpace, Unmanaged> dst_view(
      dst.get_values(), partitioner->locally_owned_size());

    const DeviceData d = data;

    Kokkos::parallel_for(
      "polydeal::DeviceDGOperator::apply",
      Kokkos::RangePolicy<ExecSpace>(0, d.n_owned_cells),
      KOKKOS_LAMBDA(const unsigned int c) {
        Number u[n_dofs];
        Number r[n_dofs];
        Number phi[n_dofs];
        Number dphi[n_dofs * dim];
        Number phi_p[n_dofs];
        Number dphi_p[n_dofs * dim];
        Number x[dim];

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            u[i] = src_view(d.dof_indices(c, i));
            r[i] = 0.;
          }

        // ----- cell term: (grad u, grad v) -----
        for (unsigned int q = 0; q < d.n_cell_q; ++q)
          {
            for (unsigned int e = 0; e < dim; ++e)
              x[e] = d.cell_q_unit(c, q, e);
            shape_values_and_gradients(d.nodes_1d, x, phi, dphi);

            // reference gradient of u
            Number gref[dim];
            for (unsigned int e = 0; e < dim; ++e)
              gref[e] = 0.;
            for (unsigned int i = 0; i < n_dofs; ++i)
              for (unsigned int e = 0; e < dim; ++e)
                gref[e] += u[i] * dphi[i * dim + e];

            // physical gradient: (grad_x u)_e = sum_d invJ(d,e) gref_d
            Number gx[dim];
            for (unsigned int e = 0; e < dim; ++e)
              {
                gx[e] = 0.;
                for (unsigned int dd = 0; dd < dim; ++dd)
                  gx[e] += d.cell_q_invJ(c, q, dd, e) * gref[dd];
              }
            // pull back: g_ref_test_d = sum_e invJ(d,e) gx_e, so that
            // grad_x phi_i . gx = sum_d dphi_i,d * g_ref_test_d
            Number       gt[dim];
            const Number w = d.cell_q_JxW(c, q);
            for (unsigned int dd = 0; dd < dim; ++dd)
              {
                gt[dd] = 0.;
                for (unsigned int e = 0; e < dim; ++e)
                  gt[dd] += d.cell_q_invJ(c, q, dd, e) * gx[e];
                gt[dd] *= w;
              }
            for (unsigned int i = 0; i < n_dofs; ++i)
              for (unsigned int dd = 0; dd < dim; ++dd)
                r[i] += dphi[i * dim + dd] * gt[dd];
          }

        // ----- face terms -----
        for (unsigned int f = 0; f < n_faces; ++f)
          {
            const unsigned int nb    = d.face_neighbor(c, f);
            const Number       sigma = d.face_sigma(c, f);
            const bool at_boundary   = (nb == numbers::invalid_unsigned_int);

            Number u_p[n_dofs];
            if (!at_boundary)
              for (unsigned int i = 0; i < n_dofs; ++i)
                u_p[i] = src_view(d.dof_indices(nb, i));

            for (unsigned int q = 0; q < d.n_face_q; ++q)
              {
                const Number w = d.face_q_JxW(c, f, q);
                Number       n[dim];
                for (unsigned int e = 0; e < dim; ++e)
                  n[e] = d.face_q_normal(c, f, q, e);

                // this side: value, physical gradient, normal derivative
                for (unsigned int e = 0; e < dim; ++e)
                  x[e] = d.face_q_unit_m(c, f, q, e);
                shape_values_and_gradients(d.nodes_1d, x, phi, dphi);

                Number u_m = 0.;
                Number gref[dim];
                for (unsigned int e = 0; e < dim; ++e)
                  gref[e] = 0.;
                for (unsigned int i = 0; i < n_dofs; ++i)
                  {
                    u_m += u[i] * phi[i];
                    for (unsigned int e = 0; e < dim; ++e)
                      gref[e] += u[i] * dphi[i * dim + e];
                  }
                Number dn_m = 0.;
                for (unsigned int e = 0; e < dim; ++e)
                  {
                    Number gx = 0.;
                    for (unsigned int dd = 0; dd < dim; ++dd)
                      gx += d.face_q_invJ_m(c, f, q, dd, e) * gref[dd];
                    dn_m += gx * n[e];
                  }

                // reference direction of n for the test functions:
                // grad_x phi_i . n = sum_d dphi_i,d * nref_d
                Number nref[dim];
                for (unsigned int dd = 0; dd < dim; ++dd)
                  {
                    nref[dd] = 0.;
                    for (unsigned int e = 0; e < dim; ++e)
                      nref[dd] += d.face_q_invJ_m(c, f, q, dd, e) * n[e];
                  }

                Number test_value, test_normal_derivative;
                if (at_boundary)
                  {
                    // from LaplaceOperatorDG::local_apply_boundary
                    test_value             = Number(2.) * sigma * u_m - dn_m;
                    test_normal_derivative = -u_m;
                  }
                else
                  {
                    // neighbour side at the same physical point
                    for (unsigned int e = 0; e < dim; ++e)
                      x[e] = d.face_q_unit_p(c, f, q, e);
                    shape_values_and_gradients(d.nodes_1d, x, phi_p, dphi_p);

                    Number u_pq = 0.;
                    Number gref_p[dim];
                    for (unsigned int e = 0; e < dim; ++e)
                      gref_p[e] = 0.;
                    for (unsigned int i = 0; i < n_dofs; ++i)
                      {
                        u_pq += u_p[i] * phi_p[i];
                        for (unsigned int e = 0; e < dim; ++e)
                          gref_p[e] += u_p[i] * dphi_p[i * dim + e];
                      }
                    Number dn_p = 0.;
                    for (unsigned int e = 0; e < dim; ++e)
                      {
                        Number gx = 0.;
                        for (unsigned int dd = 0; dd < dim; ++dd)
                          gx += d.face_q_invJ_p(c, f, q, dd, e) * gref_p[dd];
                        dn_p += gx * n[e];
                      }

                    // from LaplaceOperatorDG::local_apply_face, this side
                    const Number jump = (u_m - u_pq) * Number(0.5);
                    test_value =
                      jump * Number(2.) * sigma - (dn_m + dn_p) * Number(0.5);
                    test_normal_derivative = -jump;
                  }

                for (unsigned int i = 0; i < n_dofs; ++i)
                  {
                    Number dphi_n = 0.;
                    for (unsigned int dd = 0; dd < dim; ++dd)
                      dphi_n += dphi[i * dim + dd] * nref[dd];
                    r[i] += w * (test_value * phi[i] +
                                 test_normal_derivative * dphi_n);
                  }
              }
          }

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            const unsigned int row = d.dof_indices(c, i);
            if (add)
              dst_view(row) += r[i];
            else
              dst_view(row) = r[i];
          }
      });
    Kokkos::fence();

    if (!had_ghosts)
      src_ptr->zero_out_ghost_values();
  }



  template <int dim, int degree, typename Number>
  void
  DeviceDGOperator<dim, degree, Number>::vmult(VectorType       &dst,
                                               const VectorType &src) const
  {
    apply(dst, src, false);
  }



  template <int dim, int degree, typename Number>
  void
  DeviceDGOperator<dim, degree, Number>::vmult_add(VectorType       &dst,
                                                   const VectorType &src) const
  {
    apply(dst, src, true);
  }



  template <int dim, int degree, typename Number>
  void
  DeviceDGOperator<dim, degree, Number>::Tvmult(VectorType       &dst,
                                                const VectorType &src) const
  {
    apply(dst, src, false);
  }



  template <int dim, int degree, typename Number>
  void
  DeviceDGOperator<dim, degree, Number>::Tvmult_add(VectorType       &dst,
                                                    const VectorType &src) const
  {
    apply(dst, src, true);
  }



  template <int dim, int degree, typename Number>
  void
  DeviceDGOperator<dim, degree, Number>::get_diagonal(
    VectorType &diagonal) const
  {
    Assert(partitioner, ExcNotInitialized());
    AssertDimension(diagonal.locally_owned_size(),
                    partitioner->locally_owned_size());

    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    Kokkos::View<Number *, KokkosSpace, Unmanaged> diag_view(
      diagonal.get_values(), partitioner->locally_owned_size());

    const DeviceData d = data;

    Kokkos::parallel_for(
      "polydeal::DeviceDGOperator::get_diagonal",
      Kokkos::RangePolicy<ExecSpace>(0, d.n_owned_cells),
      KOKKOS_LAMBDA(const unsigned int c) {
        Number r[n_dofs];
        Number phi[n_dofs];
        Number dphi[n_dofs * dim];
        Number x[dim];
        for (unsigned int i = 0; i < n_dofs; ++i)
          r[i] = 0.;

        // cell: |grad phi_i|^2
        for (unsigned int q = 0; q < d.n_cell_q; ++q)
          {
            for (unsigned int e = 0; e < dim; ++e)
              x[e] = d.cell_q_unit(c, q, e);
            shape_values_and_gradients(d.nodes_1d, x, phi, dphi);
            const Number w = d.cell_q_JxW(c, q);
            for (unsigned int i = 0; i < n_dofs; ++i)
              {
                Number norm2 = 0.;
                for (unsigned int e = 0; e < dim; ++e)
                  {
                    Number gx = 0.;
                    for (unsigned int dd = 0; dd < dim; ++dd)
                      gx += d.cell_q_invJ(c, q, dd, e) * dphi[i * dim + dd];
                    norm2 += gx * gx;
                  }
                r[i] += w * norm2;
              }
          }

        // faces: this side only, u = phi_i, neighbour values zero
        for (unsigned int f = 0; f < n_faces; ++f)
          {
            const bool at_boundary =
              (d.face_neighbor(c, f) == numbers::invalid_unsigned_int);
            const Number sigma = d.face_sigma(c, f);
            for (unsigned int q = 0; q < d.n_face_q; ++q)
              {
                const Number w = d.face_q_JxW(c, f, q);
                for (unsigned int e = 0; e < dim; ++e)
                  x[e] = d.face_q_unit_m(c, f, q, e);
                shape_values_and_gradients(d.nodes_1d, x, phi, dphi);
                Number nref[dim];
                for (unsigned int dd = 0; dd < dim; ++dd)
                  {
                    nref[dd] = 0.;
                    for (unsigned int e = 0; e < dim; ++e)
                      nref[dd] += d.face_q_invJ_m(c, f, q, dd, e) *
                                  d.face_q_normal(c, f, q, e);
                  }
                for (unsigned int i = 0; i < n_dofs; ++i)
                  {
                    Number dphi_n = 0.;
                    for (unsigned int dd = 0; dd < dim; ++dd)
                      dphi_n += dphi[i * dim + dd] * nref[dd];
                    if (at_boundary)
                      r[i] += w * (Number(2.) * sigma * phi[i] * phi[i] -
                                   Number(2.) * phi[i] * dphi_n);
                    else
                      r[i] += w * (sigma * phi[i] * phi[i] - phi[i] * dphi_n);
                  }
              }
          }

        for (unsigned int i = 0; i < n_dofs; ++i)
          diag_view(d.dof_indices(c, i)) = r[i];
      });
    Kokkos::fence();
  }



  template <int dim, int degree, typename Number>
  void
  DeviceDGOperator<dim, degree, Number>::get_inverse_diagonal(
    VectorType &inverse_diagonal) const
  {
    get_diagonal(inverse_diagonal);
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    Kokkos::View<Number *, KokkosSpace, Unmanaged> diag_view(
      inverse_diagonal.get_values(), partitioner->locally_owned_size());
    Kokkos::parallel_for(
      "polydeal::DeviceDGOperator::get_inverse_diagonal",
      Kokkos::RangePolicy<ExecSpace>(0, partitioner->locally_owned_size()),
      KOKKOS_LAMBDA(const unsigned int i) {
        const Number v = diag_view(i);
        diag_view(i)   = (v == Number(0)) ? Number(1) : Number(1) / v;
      });
    Kokkos::fence();
  }



  template <int dim, int degree, typename Number>
  Number
  DeviceDGOperator<dim, degree, Number>::el(const size_type,
                                            const size_type) const
  {
    AssertThrow(
      false, ExcMessage("DeviceDGOperator does not provide element access."));
    return Number();
  }



  template <int dim, int degree, typename Number>
  std::size_t
  DeviceDGOperator<dim, degree, Number>::memory_consumption() const
  {
    std::size_t bytes = 0;
    bytes += data.nodes_1d.extent(0) * sizeof(Number);
    bytes += data.dof_indices.span() * sizeof(unsigned int);
    bytes += data.cell_q_unit.span() * sizeof(Number);
    bytes += data.cell_q_JxW.span() * sizeof(Number);
    bytes += data.cell_q_invJ.span() * sizeof(Number);
    bytes += data.face_neighbor.span() * sizeof(unsigned int);
    bytes += data.face_sigma.span() * sizeof(Number);
    bytes += data.face_q_unit_m.span() * sizeof(Number);
    bytes += data.face_q_unit_p.span() * sizeof(Number);
    bytes += data.face_q_JxW.span() * sizeof(Number);
    bytes += data.face_q_normal.span() * sizeof(Number);
    bytes += data.face_q_invJ_m.span() * sizeof(Number);
    bytes += data.face_q_invJ_p.span() * sizeof(Number);
    return bytes;
  }

} // namespace dealii

#endif
