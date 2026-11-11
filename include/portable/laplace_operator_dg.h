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

#ifndef polydeal_portable_laplace_operator_dg_h
#define polydeal_portable_laplace_operator_dg_h

#include <deal.II/base/config.h>

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/shape_info.h>

#include <Kokkos_Core.hpp>
#include <portable/dg_kokkos_kernels.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace dealii
{
  namespace PortableDG
  {
    /**
     * How the interior penalty parameter is formed from the geometry.
     */
    enum class PenaltyConvention
    {
      /**
       * The form of the original kernels: sigma = p(p+1)/2 (1/h_- + 1/h_+)
       * with h = |cell| / |face| on each side; on a Dirichlet face
       * sigma_b = p(p+1)/h.
       */
      cell_extent,

      /**
       * The form of deal.II step-59 and of
       * Utils::MatrixFreeOperators::LaplaceOperatorDG: sigma =
       * p(p+1) (|n . grad xi_-| + |n . grad xi_+|) with xi the reference
       * coordinate normal to the face, evaluated at the first quadrature
       * point; on a Dirichlet face sigma_b = 2 p(p+1) |n . grad xi_-|. On
       * Cartesian cells this is exactly twice the cell_extent value. Use it to
       * make this operator agree to rounding with the matrix assembled from
       * the host operator, and hence with Galerkin-projected coarse levels.
       */
      step59
    };



    /**
     * Symmetric interior penalty (SIP) Laplace operator for FE_DGQ on
     * quad/hex meshes, applied by the batched sum-factorised Kokkos kernels of
     * portable/dg_kokkos_kernels.h in the MemorySpace::Default memory space.
     *
     * This class is a reworking of Portable::LaplaceOperatorDG of the
     * dealiiX-portable-multigrid project. The kernels are taken over
     * unchanged; the host side has been rewritten so that the operator depends
     * only on a Mapping and a DoFHandler (no MatrixFree object, no
     * constraints, no operator base classes) and offers the same plain
     * interface as DeviceCSRMatrix and DeviceDGOperator: vmult(),
     * vmult_add(), Tvmult(), Tvmult_add(), initialize_dof_vector(),
     * get_domain_partitioner(), get_diagonal(), get_inverse_diagonal(), m(),
     * n(), el(), get_mpi_communicator().
     *
     * Discretisation (identical to the original kernels): cell integrals
     * (grad u, grad v) with QGauss(degree+1); interior faces
     * sigma [u][v] - {grad u}.n [v] - [u] {grad v}.n with n the normal of the
     * first cell and [.] = first - second; Dirichlet faces by mirroring,
     * 2 sigma_b u v - dn u v - u dn v. The penalty is selected by
     * PenaltyConvention. With all_boundaries_dirichlet = false, faces with a
     * boundary id other than zero are treated as Neumann faces as in the
     * original kernels.
     *
     * Parallel layout: the domain partitioner holds the locally owned DoFs
     * followed by the DoFs of the ghost cells that share a face with a locally
     * owned cell. Every rank evaluates the faces of its own cells, including
     * both sides of faces to ghost cells, and keeps only its own rows; no
     * compress() is needed. Vectors passed to the products are copied into
     * this layout when theirs differs.
     *
     * Limitations inherited from the kernels: no hanging nodes, standard face
     * orientation only (checked at setup), one scalar component.
     */
    template <int dim, int degree, typename Number = double>
    class LaplaceOperatorDG : public EnableObserverPointer
    {
    public:
      using value_type  = Number;
      using size_type   = types::global_dof_index;
      using KokkosSpace = typename MemorySpace::Default::kokkos_space;
      using ExecSpace   = typename KokkosSpace::execution_space;
      using VectorType =
        LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
      using HostVectorType = LinearAlgebra::distributed::Vector<Number>;
      using PartitionerPtr = std::shared_ptr<const Utilities::MPI::Partitioner>;

      static constexpr unsigned int n_q_points_1d = degree + 1;
      static constexpr unsigned int n_dofs_1d     = degree + 1;
      static constexpr unsigned int n_dofs = Utilities::pow(degree + 1, dim);
      static constexpr unsigned int n_q_points =
        Utilities::pow(n_q_points_1d, dim);
      static constexpr unsigned int n_q_points_face =
        Utilities::pow(n_q_points_1d, dim - 1);

      struct AdditionalData
      {
        PenaltyConvention penalty_convention       = PenaltyConvention::step59;
        bool              all_boundaries_dirichlet = true;
      };

      LaplaceOperatorDG() = default;

      void
      reinit(const Mapping<dim>    &mapping,
             const DoFHandler<dim> &dof_handler,
             const AdditionalData  &additional_data = AdditionalData());

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

      void
      initialize_dof_vector(VectorType &v) const;

      void
      vmult(VectorType &dst, const VectorType &src) const;

      void
      vmult_add(VectorType &dst, const VectorType &src) const;

      void
      Tvmult(VectorType &dst, const VectorType &src) const;

      void
      Tvmult_add(VectorType &dst, const VectorType &src) const;

      /**
       * The exact diagonal of the SIP operator (cell and face terms),
       * computed on the host at reinit().
       */
      void
      get_diagonal(VectorType &diagonal) const;

      void
      get_inverse_diagonal(VectorType &inverse_diagonal) const;

      Number
      el(const size_type, const size_type) const;

      unsigned int
      n_locally_owned_cells() const;

      unsigned int
      n_cells_with_ghosts() const;

      std::size_t
      memory_consumption() const;

    private:
      template <typename T>
      using View1 = Kokkos::View<T *, KokkosSpace>;

      void
      setup_cells();

      void
      setup_faces();

      void
      compute_diagonal_on_host();

      /**
       * Run the four kernels on vectors that already have the operator's
       * layout; @p dst receives the full product in its locally owned rows
       * and garbage in its ghost rows.
       */
      void
      apply_kernels(VectorType &dst, const VectorType &src) const;

      /**
       * sigma for an interior face seen from @p cell / @p face, or for a
       * Dirichlet face when @p neighbor is invalid, following the selected
       * convention. @p fe_face must be reinitialised on (cell, face).
       */
      Number
      penalty(const FEFaceValues<dim>                          &fe_face,
              const typename Triangulation<dim>::cell_iterator &cell,
              const unsigned int                                face,
              const typename Triangulation<dim>::cell_iterator &neighbor,
              const unsigned int                                neighbor_face,
              FEFaceValues<dim> &fe_face_neighbor) const;

      ObserverPointer<const Mapping<dim>>    mapping;
      ObserverPointer<const DoFHandler<dim>> dof_handler;
      AdditionalData                         additional_data;

      PartitionerPtr partitioner;
      MPI_Comm       communicator  = MPI_COMM_SELF;
      size_type      n_global_dofs = 0;
      unsigned int   n_owned_cells = 0;

      // (level, index) of locally owned cells, then of their ghost neighbours
      std::vector<std::pair<unsigned int, unsigned int>> cell_local_info;
      std::vector<std::vector<unsigned int>>             cell_level_index_map;
      std::array<unsigned int, n_dofs> lexicographic_numbering;

      // ----- device data in the layout the kernels expect -----
      View1<Number> shape_values;       // n_dofs_1d x n_q_points_1d
      View1<Number> co_shape_gradients; // n_q_points_1d x n_q_points_1d
      Kokkos::View<Number ***, KokkosSpace>      interpolate_quad_to_boundary;
      Kokkos::View<unsigned int **, KokkosSpace> dof_indices; // (dof, cell)
      View1<Number> geometric_transformation_symmetric_cell;

      Kokkos::View<Number ***, KokkosSpace> face_values_at_quads;
      Kokkos::View<Number ***, KokkosSpace> face_normal_derivatives_at_quads;

      // 0: interior faces, 1: boundary faces
      std::array<std::vector<std::array<unsigned int, 5>>, 2> face_info_cpu;
      Kokkos::Array<Kokkos::View<unsigned int *[5], KokkosSpace>, 2> face_info;
      Kokkos::View<Number *[2], KokkosSpace> jacobians_times_normal_inner_face;
      View1<Number> jacobians_times_normal_boundary_face;
      Kokkos::View<Number *[2], KokkosSpace> jxw_inner_face;
      View1<Number>                          jxw_boundary_face;
      View1<Number>                          penalty_parameters_inner_face;
      View1<Number>                          penalty_parameters_boundary_face;

      // exact diagonal, host side, locally owned entries
      HostVectorType diagonal_host;

      mutable VectorType tmp_src;
      mutable VectorType tmp_dst;
    };



    // ---------------------------- implementation ----------------------------

    template <int dim, int degree, typename Number>
    void
    LaplaceOperatorDG<dim, degree, Number>::clear()
    {
      mapping       = nullptr;
      dof_handler   = nullptr;
      partitioner   = nullptr;
      communicator  = MPI_COMM_SELF;
      n_global_dofs = 0;
      n_owned_cells = 0;
      cell_local_info.clear();
      cell_level_index_map.clear();
      face_info_cpu[0].clear();
      face_info_cpu[1].clear();
      shape_values                 = View1<Number>();
      co_shape_gradients           = View1<Number>();
      interpolate_quad_to_boundary = Kokkos::View<Number ***, KokkosSpace>();
      dof_indices = Kokkos::View<unsigned int **, KokkosSpace>();
      geometric_transformation_symmetric_cell = View1<Number>();
      face_values_at_quads = Kokkos::View<Number ***, KokkosSpace>();
      face_normal_derivatives_at_quads =
        Kokkos::View<Number ***, KokkosSpace>();
      face_info[0] = Kokkos::View<unsigned int *[5], KokkosSpace>();
      face_info[1] = Kokkos::View<unsigned int *[5], KokkosSpace>();
      jacobians_times_normal_inner_face =
        Kokkos::View<Number *[2], KokkosSpace>();
      jacobians_times_normal_boundary_face = View1<Number>();
      jxw_inner_face                = Kokkos::View<Number *[2], KokkosSpace>();
      jxw_boundary_face             = View1<Number>();
      penalty_parameters_inner_face = View1<Number>();
      penalty_parameters_boundary_face = View1<Number>();
      diagonal_host.reinit(0);
      tmp_src.reinit(0);
      tmp_dst.reinit(0);
    }



    template <int dim, int degree, typename Number>
    void
    LaplaceOperatorDG<dim, degree, Number>::reinit(
      const Mapping<dim>    &mapping_in,
      const DoFHandler<dim> &dof_handler_in,
      const AdditionalData  &additional_data_in)
    {
      clear();
      static_assert(degree >= 1,
                    "The penalty is only defined for degree >= 1.");

      mapping         = &mapping_in;
      dof_handler     = &dof_handler_in;
      additional_data = additional_data_in;

      const FiniteElement<dim> &fe = dof_handler->get_fe();
      AssertThrow(dynamic_cast<const FE_DGQ<dim> *>(&fe) != nullptr,
                  ExcMessage("PortableDG::LaplaceOperatorDG requires FE_DGQ."));
      AssertThrow(fe.degree == static_cast<unsigned int>(degree),
                  ExcMessage("The polynomial degree of the DoFHandler does "
                             "not match the template parameter."));
      AssertDimension(fe.n_dofs_per_cell(), n_dofs);

      const Triangulation<dim> &tria = dof_handler->get_triangulation();
      communicator                   = tria.get_mpi_communicator();
      n_global_dofs                  = dof_handler->n_dofs();

      // Parallel layout: owned DoFs plus the DoFs of ghost cells.
      {
        const IndexSet &owned = dof_handler->locally_owned_dofs();
        const IndexSet  relevant =
          DoFTools::extract_locally_relevant_dofs(*dof_handler);
        partitioner =
          std::make_shared<Utilities::MPI::Partitioner>(owned,
                                                        relevant,
                                                        communicator);
      }

      // 1D shape data from deal.II, in the layout the kernels expect
      // (quadrature point index running fastest).
      {
        const QGauss<1> quadrature_1d(n_q_points_1d);
        const dealii::internal::MatrixFreeFunctions::ShapeInfo<Number>
                    shape_info(quadrature_1d, fe);
        const auto &data = shape_info.get_shape_data(0, 0);

        AssertDimension(shape_info.lexicographic_numbering.size(), n_dofs);
        for (unsigned int i = 0; i < n_dofs; ++i)
          lexicographic_numbering[i] = shape_info.lexicographic_numbering[i];

        AssertDimension(data.shape_values.size(), n_dofs_1d * n_q_points_1d);
        AssertDimension(data.shape_gradients_collocation.size(),
                        n_q_points_1d * n_q_points_1d);

        shape_values = View1<Number>(
          Kokkos::view_alloc("polydeal::portable_dg::shape_values",
                             Kokkos::WithoutInitializing),
          data.shape_values.size());
        co_shape_gradients = View1<Number>(
          Kokkos::view_alloc("polydeal::portable_dg::co_shape_gradients",
                             Kokkos::WithoutInitializing),
          data.shape_gradients_collocation.size());

        using HostUnmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
        Kokkos::deep_copy(
          shape_values,
          Kokkos::View<const Number *, Kokkos::HostSpace, HostUnmanaged>(
            data.shape_values.data(), data.shape_values.size()));
        Kokkos::deep_copy(
          co_shape_gradients,
          Kokkos::View<const Number *, Kokkos::HostSpace, HostUnmanaged>(
            data.shape_gradients_collocation.data(),
            data.shape_gradients_collocation.size()));

        // Values and derivatives at 0 and 1 of the Lagrange basis on the
        // quadrature points, used to interpolate the collocation
        // representation to the faces.
        std::vector<Polynomials::Polynomial<double>> basis =
          Polynomials::generate_complete_Lagrange_basis(
            quadrature_1d.get_points());
        AssertDimension(basis.size(), n_q_points_1d);

        interpolate_quad_to_boundary = Kokkos::View<Number ***, KokkosSpace>(
          Kokkos::view_alloc(
            "polydeal::portable_dg::interpolate_quad_to_boundary",
            Kokkos::WithoutInitializing),
          2,
          basis.size(),
          2);
        auto host = Kokkos::create_mirror_view(interpolate_quad_to_boundary);
        std::vector<double> val_and_der(2);
        for (unsigned int i = 0; i < basis.size(); ++i)
          {
            basis[i].value(0., val_and_der);
            host(0, i, 0) = val_and_der[0];
            host(1, i, 0) = val_and_der[1];
            basis[i].value(1., val_and_der);
            host(0, i, 1) = val_and_der[0];
            host(1, i, 1) = val_and_der[1];
          }
        Kokkos::deep_copy(interpolate_quad_to_boundary, host);
      }

      setup_cells();
      setup_faces();

      face_values_at_quads = Kokkos::View<Number ***, KokkosSpace>(
        Kokkos::view_alloc("polydeal::portable_dg::face_values_at_quads",
                           Kokkos::WithoutInitializing),
        n_q_points_face,
        2 * dim,
        cell_local_info.size());
      face_normal_derivatives_at_quads = Kokkos::View<Number ***, KokkosSpace>(
        Kokkos::view_alloc(
          "polydeal::portable_dg::face_normal_derivatives_at_quads",
          Kokkos::WithoutInitializing),
        n_q_points_face,
        2 * dim,
        cell_local_info.size());

      compute_diagonal_on_host();
      Kokkos::fence();
    }



    template <int dim, int degree, typename Number>
    void
    LaplaceOperatorDG<dim, degree, Number>::setup_cells()
    {
      const Triangulation<dim> &tria = dof_handler->get_triangulation();

      cell_level_index_map.assign(tria.n_levels(), {});
      for (unsigned int level = 0; level < tria.n_levels(); ++level)
        cell_level_index_map[level].assign(tria.n_raw_cells(level),
                                           numbers::invalid_unsigned_int);

      // Locally owned cells first, then the ghost cells across their faces.
      cell_local_info.clear();
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell_level_index_map[cell->level()][cell->index()] =
              cell_local_info.size();
            cell_local_info.emplace_back(cell->level(), cell->index());
          }
      n_owned_cells = cell_local_info.size();

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int f : cell->face_indices())
            if (!cell->at_boundary(f))
              {
                const auto neighbor = cell->neighbor(f);
                AssertThrow(!neighbor->is_artificial(),
                            ExcMessage("A neighbour of a locally owned cell "
                                       "must be locally owned or ghosted."));
                AssertThrow(neighbor->is_active() &&
                              !cell->neighbor_is_coarser(f),
                            ExcMessage("Hanging nodes are not supported."));
                unsigned int &row =
                  cell_level_index_map[neighbor->level()][neighbor->index()];
                if (row == numbers::invalid_unsigned_int)
                  {
                    row = cell_local_info.size();
                    cell_local_info.emplace_back(neighbor->level(),
                                                 neighbor->index());
                  }
              }

      const unsigned int n_cells = cell_local_info.size();

      // DoF indices in the partitioner's local numbering, lexicographic
      dof_indices = Kokkos::View<unsigned int **, KokkosSpace>(
        Kokkos::view_alloc("polydeal::portable_dg::dof_indices",
                           Kokkos::WithoutInitializing),
        n_dofs,
        n_cells);
      {
        auto host = Kokkos::create_mirror_view(dof_indices);
        std::vector<types::global_dof_index> local_dofs(n_dofs);
        for (unsigned int c = 0; c < n_cells; ++c)
          {
            const typename DoFHandler<dim>::active_cell_iterator cell(
              &tria,
              cell_local_info[c].first,
              cell_local_info[c].second,
              &*dof_handler);
            cell->get_dof_indices(local_dofs);
            for (unsigned int i = 0; i < n_dofs; ++i)
              host(i, c) = partitioner->global_to_local(
                local_dofs[lexicographic_numbering[i]]);
          }
        Kokkos::deep_copy(dof_indices, host);
      }

      // JxW J^{-1} J^{-T} per quadrature point, symmetric part only
      {
        constexpr int symmetric_tensor_dim = (dim * (dim + 1)) / 2;
        geometric_transformation_symmetric_cell =
          View1<Number>(Kokkos::view_alloc("polydeal::portable_dg::geometry",
                                           Kokkos::WithoutInitializing),
                        static_cast<std::size_t>(n_cells) *
                          symmetric_tensor_dim * n_q_points);
        auto host =
          Kokkos::create_mirror_view(geometric_transformation_symmetric_cell);

        const QGauss<dim> quadrature(n_q_points_1d);
        FEValues<dim>     fe_values(*mapping,
                                dof_handler->get_fe(),
                                quadrature,
                                update_JxW_values | update_jacobians);
        for (unsigned int c = 0; c < n_cells; ++c)
          {
            const typename Triangulation<dim>::active_cell_iterator cell(
              &tria, cell_local_info[c].first, cell_local_info[c].second);
            fe_values.reinit(cell);
            const std::size_t base =
              static_cast<std::size_t>(c) * symmetric_tensor_dim * n_q_points;
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                const Tensor<2, dim, double> inv_jacobian(
                  fe_values.jacobian(q).covariant_form());
                const double jxw = fe_values.JxW(q);
                int          idx = 0;
                for (unsigned int d1 = 0; d1 < dim; ++d1)
                  for (unsigned int d2 = d1; d2 < dim; ++d2, ++idx)
                    {
                      double sum = 0;
                      for (unsigned int k = 0; k < dim; ++k)
                        sum += inv_jacobian[k][d1] * inv_jacobian[k][d2];
                      host[base + idx * n_q_points + q] = jxw * sum;
                    }
              }
          }
        Kokkos::deep_copy(geometric_transformation_symmetric_cell, host);
      }
    }



    template <int dim, int degree, typename Number>
    Number
    LaplaceOperatorDG<dim, degree, Number>::penalty(
      const FEFaceValues<dim>                          &fe_face,
      const typename Triangulation<dim>::cell_iterator &cell,
      const unsigned int                                face,
      const typename Triangulation<dim>::cell_iterator &neighbor,
      const unsigned int                                neighbor_face,
      FEFaceValues<dim>                                &fe_face_neighbor) const
    {
      const double factor      = degree * (degree + 1.);
      const bool   at_boundary = (neighbor.state() != IteratorState::valid);

      if (additional_data.penalty_convention == PenaltyConvention::cell_extent)
        {
          const double extent_m = cell->measure() / cell->face(face)->measure();
          if (at_boundary)
            return factor / extent_m;
          const double extent_p =
            neighbor->measure() / neighbor->face(neighbor_face)->measure();
          return factor * 0.5 * (1. / extent_m + 1. / extent_p);
        }

      // step-59: |n . grad xi_k| at the first quadrature point, with k the
      // reference direction normal to the face (row k of J^{-1}).
      const Tensor<1, dim>         n = fe_face.normal_vector(0);
      const Tensor<2, dim, double> inv_jac_m(
        fe_face.jacobian(0).covariant_form());
      const Tensor<1, dim, double> jac_x_n_m = n * inv_jac_m;
      const double                 pen_m     = std::abs(jac_x_n_m[face / 2]);
      if (at_boundary)
        return 2. * pen_m * factor;

      fe_face_neighbor.reinit(neighbor, neighbor_face);
      const Tensor<2, dim, double> inv_jac_p(
        fe_face_neighbor.jacobian(0).covariant_form());
      const Tensor<1, dim, double> jac_x_n_p = n * inv_jac_p;
      const double pen_p = std::abs(jac_x_n_p[neighbor_face / 2]);
      return (pen_m + pen_p) * factor;
    }



    template <int dim, int degree, typename Number>
    void
    LaplaceOperatorDG<dim, degree, Number>::setup_faces()
    {
      const Triangulation<dim> &tria = dof_handler->get_triangulation();

      // Face lists. Entries: 0 interior cell, 1 exterior cell (invalid for
      // boundary faces), 2 interior face number, 3 exterior face number or
      // boundary id, 4 orientation (always standard here).
      face_info_cpu[0].clear();
      face_info_cpu[1].clear();
      {
        std::vector<unsigned char> visited_face(tria.n_raw_faces(), 0);
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            for (const unsigned int f : cell->face_indices())
              {
                const unsigned int face_index = cell->face(f)->index();
                if (visited_face[face_index])
                  continue;
                visited_face[face_index] = 1;

                std::array<unsigned int, 5> info;
                info[0] = cell_level_index_map[cell->level()][cell->index()];
                info[2] = f;
                info[4] = 0;
                if (cell->at_boundary(f))
                  {
                    info[1] = numbers::invalid_unsigned_int;
                    info[3] = additional_data.all_boundaries_dirichlet ?
                                0 :
                                cell->face(f)->boundary_id();
                    face_info_cpu[1].push_back(info);
                  }
                else
                  {
                    const auto neighbor = cell->neighbor(f);
                    info[1] = cell_level_index_map[neighbor->level()]
                                                  [neighbor->index()];
                    info[3] = cell->neighbor_of_neighbor(f);
                    face_info_cpu[0].push_back(info);
                  }
              }

        const std::string names[2] = {"polydeal::portable_dg::inner_faces",
                                      "polydeal::portable_dg::boundary_faces"};
        for (unsigned int i = 0; i < 2; ++i)
          {
            const unsigned int n_faces = face_info_cpu[i].size();
            face_info[i] = Kokkos::View<unsigned int *[5], KokkosSpace>(
              Kokkos::view_alloc(names[i], Kokkos::WithoutInitializing),
              n_faces);
            if (n_faces > 0)
              {
                auto host = Kokkos::create_mirror_view(face_info[i]);
                for (unsigned int f = 0; f < n_faces; ++f)
                  for (unsigned int k = 0; k < 5; ++k)
                    host(f, k) = face_info_cpu[i][f][k];
                Kokkos::deep_copy(face_info[i], host);
              }
          }
      }

      const unsigned int n_inner    = face_info_cpu[0].size();
      const unsigned int n_boundary = face_info_cpu[1].size();

      jacobians_times_normal_inner_face =
        Kokkos::View<Number *[2], KokkosSpace>(
          Kokkos::view_alloc("polydeal::portable_dg::jac_x_n_inner",
                             Kokkos::WithoutInitializing),
          static_cast<std::size_t>(n_inner) * dim * n_q_points_face);
      jxw_inner_face = Kokkos::View<Number *[2], KokkosSpace>(
        Kokkos::view_alloc("polydeal::portable_dg::jxw_inner",
                           Kokkos::WithoutInitializing),
        static_cast<std::size_t>(n_inner) * n_q_points_face);
      penalty_parameters_inner_face =
        View1<Number>(Kokkos::view_alloc("polydeal::portable_dg::penalty_inner",
                                         Kokkos::WithoutInitializing),
                      n_inner);
      jacobians_times_normal_boundary_face = View1<Number>(
        Kokkos::view_alloc("polydeal::portable_dg::jac_x_n_boundary",
                           Kokkos::WithoutInitializing),
        static_cast<std::size_t>(n_boundary) * dim * n_q_points_face);
      jxw_boundary_face =
        View1<Number>(Kokkos::view_alloc("polydeal::portable_dg::jxw_boundary",
                                         Kokkos::WithoutInitializing),
                      static_cast<std::size_t>(n_boundary) * n_q_points_face);
      penalty_parameters_boundary_face = View1<Number>(
        Kokkos::view_alloc("polydeal::portable_dg::penalty_boundary",
                           Kokkos::WithoutInitializing),
        n_boundary);

      auto h_jac_inner =
        Kokkos::create_mirror_view(jacobians_times_normal_inner_face);
      auto h_jac_bdry =
        Kokkos::create_mirror_view(jacobians_times_normal_boundary_face);
      auto h_jxw_inner = Kokkos::create_mirror_view(jxw_inner_face);
      auto h_jxw_bdry  = Kokkos::create_mirror_view(jxw_boundary_face);
      auto h_pen_inner =
        Kokkos::create_mirror_view(penalty_parameters_inner_face);
      auto h_pen_bdry =
        Kokkos::create_mirror_view(penalty_parameters_boundary_face);

      const QGauss<dim - 1> face_quadrature(n_q_points_1d);
      const UpdateFlags     flags = update_JxW_values | update_normal_vectors |
                                update_jacobians | update_quadrature_points;
      FEFaceValues<dim> fe_face(*mapping,
                                dof_handler->get_fe(),
                                face_quadrature,
                                flags);
      FEFaceValues<dim> fe_face_neighbor(*mapping,
                                         dof_handler->get_fe(),
                                         face_quadrature,
                                         flags);

      // The kernels number the quadrature points of a face lexicographically
      // in the cell's reference coordinates (lower tangential direction
      // fastest). deal.II's face quadrature does the same except on the
      // y-normal faces 2 and 3 of a hexahedron, whose face coordinates are
      // (z, x): remap those to the kernels' order.
      const auto kernel_face_q = [](const unsigned int face_no,
                                    const unsigned int q) -> unsigned int {
        if (dim == 3 && face_no / 2 == 1)
          return (q % n_q_points_1d) * n_q_points_1d + q / n_q_points_1d;
        return q;
      };

      const typename Triangulation<dim>::cell_iterator invalid_cell;

      for (unsigned int f = 0; f < n_inner; ++f)
        {
          const unsigned int f_minus = face_info_cpu[0][f][2];
          const unsigned int f_plus  = face_info_cpu[0][f][3];
          const typename Triangulation<dim>::cell_iterator cell_minus(
            &tria,
            cell_local_info[face_info_cpu[0][f][0]].first,
            cell_local_info[face_info_cpu[0][f][0]].second);
          const typename Triangulation<dim>::cell_iterator cell_plus(
            &tria,
            cell_local_info[face_info_cpu[0][f][1]].first,
            cell_local_info[face_info_cpu[0][f][1]].second);

          fe_face.reinit(cell_minus, f_minus);
          h_pen_inner(f) = penalty(
            fe_face, cell_minus, f_minus, cell_plus, f_plus, fe_face_neighbor);
          fe_face_neighbor.reinit(cell_plus, f_plus);

          for (unsigned int q = 0; q < n_q_points_face; ++q)
            {
              // The kernels pair quadrature point q of both sides; this holds
              // for faces in standard orientation only.
              AssertThrow(fe_face.quadrature_point(q).distance(
                            fe_face_neighbor.quadrature_point(q)) <
                            1e-10 * cell_minus->diameter(),
                          ExcMessage(
                            "The quadrature points of the two sides of a face "
                            "do not coincide: faces with non-standard "
                            "orientation are not supported."));

              const Tensor<1, dim>         n_minus = fe_face.normal_vector(q);
              const Tensor<2, dim, double> inv_jac_minus(
                fe_face.jacobian(q).covariant_form());
              const Tensor<2, dim, double> inv_jac_plus(
                fe_face_neighbor.jacobian(q).covariant_form());
              // n . grad xi_d on both sides, with the normal of the first cell
              const Tensor<1, dim, double> jac_x_n_minus =
                n_minus * inv_jac_minus;
              const Tensor<1, dim, double> jac_x_n_plus =
                n_minus * inv_jac_plus;

              const unsigned int q_k = kernel_face_q(f_minus, q);
              h_jxw_inner(f * n_q_points_face + q_k, 0) = fe_face.JxW(q);
              h_jxw_inner(f * n_q_points_face + q_k, 1) =
                fe_face_neighbor.JxW(q);
              for (unsigned int d = 0; d < dim; ++d)
                {
                  const std::size_t idx =
                    static_cast<std::size_t>(f) * dim * n_q_points_face +
                    d * n_q_points_face + q_k;
                  h_jac_inner(idx, 0) = jac_x_n_minus[d];
                  h_jac_inner(idx, 1) = jac_x_n_plus[d];
                }
            }
        }

      for (unsigned int f = 0; f < n_boundary; ++f)
        {
          const unsigned int face = face_info_cpu[1][f][2];
          const typename Triangulation<dim>::cell_iterator cell(
            &tria,
            cell_local_info[face_info_cpu[1][f][0]].first,
            cell_local_info[face_info_cpu[1][f][0]].second);
          fe_face.reinit(cell, face);
          h_pen_bdry(f) = penalty(fe_face,
                                  cell,
                                  face,
                                  invalid_cell,
                                  numbers::invalid_unsigned_int,
                                  fe_face_neighbor);

          for (unsigned int q = 0; q < n_q_points_face; ++q)
            {
              const Tensor<1, dim>         n = fe_face.normal_vector(q);
              const Tensor<2, dim, double> inv_jac(
                fe_face.jacobian(q).covariant_form());
              const Tensor<1, dim, double> jac_x_n  = n * inv_jac;
              const unsigned int           q_k      = kernel_face_q(face, q);
              h_jxw_bdry(f * n_q_points_face + q_k) = fe_face.JxW(q);
              for (unsigned int d = 0; d < dim; ++d)
                h_jac_bdry(static_cast<std::size_t>(f) * dim * n_q_points_face +
                           d * n_q_points_face + q_k) = jac_x_n[d];
            }
        }

      Kokkos::deep_copy(jacobians_times_normal_inner_face, h_jac_inner);
      Kokkos::deep_copy(jacobians_times_normal_boundary_face, h_jac_bdry);
      Kokkos::deep_copy(jxw_inner_face, h_jxw_inner);
      Kokkos::deep_copy(jxw_boundary_face, h_jxw_bdry);
      Kokkos::deep_copy(penalty_parameters_inner_face, h_pen_inner);
      Kokkos::deep_copy(penalty_parameters_boundary_face, h_pen_bdry);
    }



    template <int dim, int degree, typename Number>
    void
    LaplaceOperatorDG<dim, degree, Number>::compute_diagonal_on_host()
    {
      // diag_i = (grad phi_i, grad phi_i)_K
      //        + sum over interior faces of K: int sigma phi_i^2 - phi_i dn
      //        phi_i
      //        + sum over Dirichlet faces:     int 2 sigma_b phi_i^2 - 2 phi_i
      //        dn phi_i
      //        + sum over Neumann faces:       int - phi_i dn phi_i
      // which is what the kernels produce for u = phi_i with the neighbour
      // values zero.
      const FiniteElement<dim> &fe = dof_handler->get_fe();

      diagonal_host.reinit(partitioner);

      const QGauss<dim>     quadrature(n_q_points_1d);
      const QGauss<dim - 1> face_quadrature(n_q_points_1d);
      FEValues<dim>         fe_values(*mapping,
                              fe,
                              quadrature,
                              update_gradients | update_JxW_values);
      const UpdateFlags     face_flags = update_values | update_gradients |
                                     update_JxW_values | update_normal_vectors |
                                     update_jacobians;
      FEFaceValues<dim> fe_face(*mapping, fe, face_quadrature, face_flags);
      FEFaceValues<dim> fe_face_neighbor(*mapping,
                                         fe,
                                         face_quadrature,
                                         face_flags);

      std::vector<types::global_dof_index>             local_dofs(n_dofs);
      std::vector<double>                              local_diag(n_dofs);
      const typename Triangulation<dim>::cell_iterator invalid_cell;

      for (const auto &cell : dof_handler->active_cell_iterators())
        if (cell->is_locally_owned())
          {
            std::fill(local_diag.begin(), local_diag.end(), 0.);

            fe_values.reinit(cell);
            for (unsigned int q = 0; q < n_q_points; ++q)
              for (unsigned int i = 0; i < n_dofs; ++i)
                local_diag[i] +=
                  fe_values.shape_grad(i, q).norm_square() * fe_values.JxW(q);

            for (const unsigned int f : cell->face_indices())
              {
                fe_face.reinit(cell, f);
                const bool at_boundary = cell->at_boundary(f);
                bool       dirichlet   = true;
                double     sigma;
                if (at_boundary)
                  {
                    dirichlet = additional_data.all_boundaries_dirichlet ||
                                cell->face(f)->boundary_id() == 0;
                    sigma = penalty(fe_face,
                                    cell,
                                    f,
                                    invalid_cell,
                                    numbers::invalid_unsigned_int,
                                    fe_face_neighbor);
                  }
                else
                  sigma = penalty(fe_face,
                                  cell,
                                  f,
                                  cell->neighbor(f),
                                  cell->neighbor_of_neighbor(f),
                                  fe_face_neighbor);

                for (unsigned int q = 0; q < n_q_points_face; ++q)
                  {
                    const double w = fe_face.JxW(q);
                    for (unsigned int i = 0; i < n_dofs; ++i)
                      {
                        const double v = fe_face.shape_value(i, q);
                        const double dn =
                          fe_face.shape_grad(i, q) * fe_face.normal_vector(q);
                        if (!at_boundary)
                          local_diag[i] += w * (sigma * v * v - v * dn);
                        else if (dirichlet)
                          local_diag[i] +=
                            w * (2. * sigma * v * v - 2. * v * dn);
                        else
                          local_diag[i] += w * (-v * dn);
                      }
                  }
              }

            cell->get_dof_indices(local_dofs);
            for (unsigned int i = 0; i < n_dofs; ++i)
              diagonal_host(local_dofs[i]) = local_diag[i];
          }
    }



    template <int dim, int degree, typename Number>
    void
    LaplaceOperatorDG<dim, degree, Number>::apply_kernels(
      VectorType       &dst,
      const VectorType &src) const
    {
      const unsigned int n_cells    = cell_local_info.size();
      const unsigned int n_inner    = face_info_cpu[0].size();
      const unsigned int n_boundary = face_info_cpu[1].size();
      const unsigned int layout_size =
        partitioner->locally_owned_size() + partitioner->n_ghost_indices();

      AssertDimension(static_cast<std::size_t>(n_cells) * n_dofs, layout_size);

      // Unmanaged views over the vectors' storage, owned + ghost entries.
      const View1<Number> src_device(src.get_values(), layout_size);
      View1<Number>       dst_device(dst.get_values(), layout_size);

      // Launch configuration. The kernels decide batch sizes from a shared
      // memory budget. On host backends a team is a single thread; the
      // league is sized automatically so that batches spread over threads.
      constexpr bool host_backend =
        std::is_same_v<Kokkos::DefaultExecutionSpace,
                       Kokkos::DefaultHostExecutionSpace>;
      const unsigned int n_per_batch = numbers::invalid_unsigned_int;
      const unsigned int n_blocks    = numbers::invalid_unsigned_int;
      const unsigned int threads_per_block =
        host_backend ? 1u : numbers::invalid_unsigned_int;

      if (n_cells == 0)
        return;

      kernels::compute_cell<dim, n_dofs_1d, n_q_points_1d, Number>(
        shape_values,
        co_shape_gradients,
        geometric_transformation_symmetric_cell,
        src_device,
        dst_device,
        interpolate_quad_to_boundary,
        face_values_at_quads,
        face_normal_derivatives_at_quads,
        dof_indices,
        n_cells,
        n_per_batch,
        n_blocks,
        threads_per_block);
      Kokkos::fence();

      if (n_inner > 0)
        {
          kernels::compute_inner_faces<dim, n_dofs_1d, n_q_points_1d, Number>(
            co_shape_gradients,
            jacobians_times_normal_inner_face,
            jxw_inner_face,
            penalty_parameters_inner_face,
            face_values_at_quads,
            face_normal_derivatives_at_quads,
            face_info[0],
            n_inner,
            n_per_batch,
            n_blocks,
            threads_per_block);
          Kokkos::fence();
        }

      if (n_boundary > 0)
        {
          kernels::
            compute_boundary_faces<dim, n_dofs_1d, n_q_points_1d, Number>(
              co_shape_gradients,
              jacobians_times_normal_boundary_face,
              jxw_boundary_face,
              penalty_parameters_boundary_face,
              face_values_at_quads,
              face_normal_derivatives_at_quads,
              face_info[1],
              n_boundary,
              n_per_batch,
              n_blocks,
              threads_per_block);
          Kokkos::fence();
        }

      kernels::distribute_face_to_global<dim, n_dofs_1d, n_q_points_1d, Number>(
        shape_values,
        dst_device,
        interpolate_quad_to_boundary,
        face_values_at_quads,
        face_normal_derivatives_at_quads,
        dof_indices,
        n_cells,
        n_per_batch,
        n_blocks,
        threads_per_block);
      Kokkos::fence();
    }



    template <int dim, int degree, typename Number>
    void
    LaplaceOperatorDG<dim, degree, Number>::vmult(VectorType       &dst,
                                                  const VectorType &src) const
    {
      Assert(partitioner, ExcNotInitialized());
      AssertDimension(src.locally_owned_size(),
                      partitioner->locally_owned_size());
      AssertDimension(dst.locally_owned_size(),
                      partitioner->locally_owned_size());

      // Source in the operator's layout
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

      // Destination in the operator's layout: the kernels write ghost rows.
      VectorType *dst_ptr = &dst;
      if (dst.get_partitioner().get() != partitioner.get() &&
          !dst.get_partitioner()->is_compatible(*partitioner))
        {
          if (tmp_dst.get_partitioner().get() != partitioner.get())
            tmp_dst.reinit(partitioner);
          dst_ptr = &tmp_dst;
        }

      apply_kernels(*dst_ptr, *src_ptr);

      dst_ptr->zero_out_ghost_values();
      if (dst_ptr != &dst)
        dst.copy_locally_owned_data_from(*dst_ptr);

      if (!had_ghosts)
        src_ptr->zero_out_ghost_values();
    }



    template <int dim, int degree, typename Number>
    void
    LaplaceOperatorDG<dim, degree, Number>::vmult_add(
      VectorType       &dst,
      const VectorType &src) const
    {
      VectorType product;
      product.reinit(partitioner);
      vmult(product, src);
      dst.add(Number(1.), product);
    }



    template <int dim, int degree, typename Number>
    void
    LaplaceOperatorDG<dim, degree, Number>::Tvmult(VectorType       &dst,
                                                   const VectorType &src) const
    {
      vmult(dst, src);
    }



    template <int dim, int degree, typename Number>
    void
    LaplaceOperatorDG<dim, degree, Number>::Tvmult_add(
      VectorType       &dst,
      const VectorType &src) const
    {
      vmult_add(dst, src);
    }



    template <int dim, int degree, typename Number>
    void
    LaplaceOperatorDG<dim, degree, Number>::get_diagonal(
      VectorType &diagonal) const
    {
      Assert(partitioner, ExcNotInitialized());
      AssertDimension(diagonal.locally_owned_size(),
                      partitioner->locally_owned_size());
      if (diagonal.get_partitioner().get() == partitioner.get() ||
          diagonal.get_partitioner()->is_compatible(*partitioner))
        diagonal.import_elements(diagonal_host, VectorOperation::insert);
      else
        {
          VectorType in_layout;
          in_layout.reinit(partitioner);
          in_layout.import_elements(diagonal_host, VectorOperation::insert);
          diagonal.copy_locally_owned_data_from(in_layout);
        }
    }



    template <int dim, int degree, typename Number>
    void
    LaplaceOperatorDG<dim, degree, Number>::get_inverse_diagonal(
      VectorType &inverse_diagonal) const
    {
      HostVectorType inverse(diagonal_host);
      for (unsigned int i = 0; i < inverse.locally_owned_size(); ++i)
        inverse.local_element(i) =
          (diagonal_host.local_element(i) == Number(0)) ?
            Number(1) :
            Number(1) / diagonal_host.local_element(i);
      VectorType in_layout;
      in_layout.reinit(partitioner);
      in_layout.import_elements(inverse, VectorOperation::insert);
      if (inverse_diagonal.get_partitioner().get() == partitioner.get() ||
          inverse_diagonal.get_partitioner()->is_compatible(*partitioner))
        inverse_diagonal = in_layout;
      else
        inverse_diagonal.copy_locally_owned_data_from(in_layout);
    }



    template <int dim, int degree, typename Number>
    inline typename LaplaceOperatorDG<dim, degree, Number>::size_type
    LaplaceOperatorDG<dim, degree, Number>::m() const
    {
      return n_global_dofs;
    }



    template <int dim, int degree, typename Number>
    inline typename LaplaceOperatorDG<dim, degree, Number>::size_type
    LaplaceOperatorDG<dim, degree, Number>::n() const
    {
      return n_global_dofs;
    }



    template <int dim, int degree, typename Number>
    inline MPI_Comm
    LaplaceOperatorDG<dim, degree, Number>::get_mpi_communicator() const
    {
      return communicator;
    }



    template <int dim, int degree, typename Number>
    inline const typename LaplaceOperatorDG<dim, degree, Number>::
      PartitionerPtr &
      LaplaceOperatorDG<dim, degree, Number>::get_domain_partitioner() const
    {
      return partitioner;
    }



    template <int dim, int degree, typename Number>
    void
    LaplaceOperatorDG<dim, degree, Number>::initialize_dof_vector(
      VectorType &v) const
    {
      Assert(partitioner, ExcNotInitialized());
      v.reinit(partitioner);
    }



    template <int dim, int degree, typename Number>
    Number
    LaplaceOperatorDG<dim, degree, Number>::el(const size_type,
                                               const size_type) const
    {
      AssertThrow(false,
                  ExcMessage("PortableDG::LaplaceOperatorDG does not provide "
                             "element access."));
      return Number();
    }



    template <int dim, int degree, typename Number>
    inline unsigned int
    LaplaceOperatorDG<dim, degree, Number>::n_locally_owned_cells() const
    {
      return n_owned_cells;
    }



    template <int dim, int degree, typename Number>
    inline unsigned int
    LaplaceOperatorDG<dim, degree, Number>::n_cells_with_ghosts() const
    {
      return cell_local_info.size();
    }



    template <int dim, int degree, typename Number>
    std::size_t
    LaplaceOperatorDG<dim, degree, Number>::memory_consumption() const
    {
      std::size_t bytes = 0;
      bytes += shape_values.span() * sizeof(Number);
      bytes += co_shape_gradients.span() * sizeof(Number);
      bytes += interpolate_quad_to_boundary.span() * sizeof(Number);
      bytes += dof_indices.span() * sizeof(unsigned int);
      bytes += geometric_transformation_symmetric_cell.span() * sizeof(Number);
      bytes += face_values_at_quads.span() * sizeof(Number);
      bytes += face_normal_derivatives_at_quads.span() * sizeof(Number);
      bytes += face_info[0].span() * sizeof(unsigned int);
      bytes += face_info[1].span() * sizeof(unsigned int);
      bytes += jacobians_times_normal_inner_face.span() * sizeof(Number);
      bytes += jacobians_times_normal_boundary_face.span() * sizeof(Number);
      bytes += jxw_inner_face.span() * sizeof(Number);
      bytes += jxw_boundary_face.span() * sizeof(Number);
      bytes += penalty_parameters_inner_face.span() * sizeof(Number);
      bytes += penalty_parameters_boundary_face.span() * sizeof(Number);
      return bytes;
    }

  } // namespace PortableDG
} // namespace dealii

#endif
