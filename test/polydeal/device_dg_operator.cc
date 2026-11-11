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


// Check the two device matrix-free SIP-DG operators against the host
// matrix-free operator of utils.h and against the Trilinos matrix assembled
// from it: products, accumulating products, the copy path for vectors in a
// foreign layout, and the diagonal. Cartesian and randomly distorted meshes
// in 2D and 3D, at degrees 1 and 2.
//
//  - DeviceDGOperator: cell-centric kernel evaluating the basis at stored
//    reference points;
//  - PortableDG::LaplaceOperatorDG: batched sum-factorised kernels copied
//    from dealiiX-portable-multigrid, with the step-59 penalty convention.
//    Its original cell_extent convention is checked for symmetry, positivity
//    and the known relation to the step-59 operator on Cartesian meshes.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <agglomeration_handler.h>
#include <device_dg_operator.h>
#include <portable/laplace_operator_dg.h>
#include <utils.h>

#include <cmath>

using namespace dealii;

using HostVector = LinearAlgebra::distributed::Vector<double>;
using DeviceVector =
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;



void
fill_test_vector(HostVector &v, const double shift)
{
  for (const types::global_dof_index i : v.locally_owned_elements())
    v(i) = std::sin(0.37 * i + shift) + 0.1 * (i % 7);
}



// Relative difference of the locally owned parts, independent of ghost
// layouts.
double
relative_difference(const HostVector &reference, const DeviceVector &device)
{
  HostVector back(device.get_partitioner());
  back.import_elements(device, VectorOperation::insert);
  HostVector diff(reference);
  diff.copy_locally_owned_data_from(back);
  diff -= reference;
  const double ref_norm = reference.l2_norm();
  return diff.l2_norm() / (ref_norm > 0 ? ref_norm : 1.);
}



std::string
verdict(const double rel)
{
  return rel < 1e-11 ? "OK" : "FAIL (" + std::to_string(rel) + ")";
}



template <int dim, int degree>
void
run(const MPI_Comm comm, const unsigned int n_refinements, const bool distort)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(comm) == 0);
  const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(comm);
  pcout << "*** dim=" << dim << ", degree=" << degree
        << (distort ? ", distorted" : ", Cartesian") << ", ranks=" << n_ranks
        << " ***" << std::endl;

  Triangulation<dim> serial_tria;
  GridGenerator::hyper_cube(serial_tria, 0., 1.);
  serial_tria.refine_global(n_refinements);
  if (distort)
    GridTools::distort_random(0.2, serial_tria, true);
  GridTools::partition_triangulation(n_ranks, serial_tria);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial_tria, comm);
  parallel::fullydistributed::Triangulation<dim> tria(comm);
  tria.create_triangulation(description);

  FE_DGQ<dim>     fe_dg(degree);
  MappingQ1<dim>  mapping;
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_dg);
  pcout << "Cells: " << tria.n_global_active_cells()
        << ", DoFs: " << dof_handler.n_dofs() << std::endl;

  // Host references
  Utils::MatrixFreeOperators::LaplaceOperatorDG<dim, degree, degree + 1, 1>
    host_operator;
  host_operator.reinit(mapping, dof_handler);
  const TrilinosWrappers::SparseMatrix &matrix =
    host_operator.get_system_matrix();

  HostVector x, y_mf;
  host_operator.initialize_dof_vector(x);
  host_operator.initialize_dof_vector(y_mf);
  fill_test_vector(x, 0.1);
  host_operator.vmult(y_mf, x);

  HostVector x_owned(dof_handler.locally_owned_dofs(), comm);
  HostVector y_mat(dof_handler.locally_owned_dofs(), comm);
  x_owned.copy_locally_owned_data_from(x);
  matrix.vmult(y_mat, x_owned);

  // Device operator
  DeviceDGOperator<dim, degree> device_operator;
  device_operator.reinit(mapping, dof_handler);

  const std::size_t device_bytes =
    Utilities::MPI::sum(device_operator.memory_consumption(), comm);
  const std::size_t matrix_bytes =
    matrix.n_nonzero_elements() * (sizeof(double) + sizeof(int));
  pcout << "Device operator data: " << device_bytes << " bytes; assembled "
        << "matrix values + indices: " << matrix_bytes << " bytes" << std::endl;

  HostVector x_layout(device_operator.get_domain_partitioner());
  x_layout.copy_locally_owned_data_from(x);
  DeviceVector x_dev, y_dev;
  device_operator.initialize_dof_vector(x_dev);
  device_operator.initialize_dof_vector(y_dev);
  x_dev.import_elements(x_layout, VectorOperation::insert);

  device_operator.vmult(y_dev, x_dev);
  pcout << "vmult vs host matrix-free: "
        << verdict(relative_difference(y_mf, y_dev)) << std::endl;
  pcout << "vmult vs assembled matrix: "
        << verdict(relative_difference(y_mat, y_dev)) << std::endl;

  // Source in a foreign layout (locally owned only) and vmult_add
  {
    DeviceVector x_plain(dof_handler.locally_owned_dofs(), comm);
    x_plain.import_elements(x_owned, VectorOperation::insert);
    DeviceVector y2(dof_handler.locally_owned_dofs(), comm);
    device_operator.vmult(y2, x_plain);
    pcout << "vmult with plain source: "
          << verdict(relative_difference(y_mat, y2)) << std::endl;

    HostVector y_twice(y_mat);
    y_twice *= 2.;
    device_operator.vmult_add(y2, x_plain);
    pcout << "vmult_add: " << verdict(relative_difference(y_twice, y2))
          << std::endl;

    device_operator.Tvmult(y2, x_plain);
    pcout << "Tvmult: " << verdict(relative_difference(y_mat, y2)) << std::endl;
  }

  // Diagonal
  HostVector diag(dof_handler.locally_owned_dofs(), comm);
  for (const types::global_dof_index i : dof_handler.locally_owned_dofs())
    diag(i) = matrix.diag_element(i);
  HostVector inv_diag(diag);
  for (const types::global_dof_index i : dof_handler.locally_owned_dofs())
    inv_diag(i) = 1. / diag(i);
  {
    DeviceVector diag_dev(dof_handler.locally_owned_dofs(), comm);
    device_operator.get_diagonal(diag_dev);
    pcout << "diagonal: " << verdict(relative_difference(diag, diag_dev))
          << std::endl;
    device_operator.get_inverse_diagonal(diag_dev);
    pcout << "inverse diagonal: "
          << verdict(relative_difference(inv_diag, diag_dev)) << std::endl;
  }

  // ---------------------------------------------------------------------
  // Batched operator, step-59 penalty: must agree with the assembled matrix
  // ---------------------------------------------------------------------
  pcout << "-- PortableDG::LaplaceOperatorDG, step59 penalty --" << std::endl;
  PortableDG::LaplaceOperatorDG<dim, degree> batched;
  {
    typename PortableDG::LaplaceOperatorDG<dim, degree>::AdditionalData data;
    data.penalty_convention = PortableDG::PenaltyConvention::step59;
    batched.reinit(mapping, dof_handler, data);
  }
  pcout << "cells with ghosts: " << batched.n_cells_with_ghosts()
        << ", owned: " << batched.n_locally_owned_cells() << "; device data "
        << Utilities::MPI::sum(batched.memory_consumption(), comm) << " bytes"
        << std::endl;

  DeviceVector xb, yb;
  batched.initialize_dof_vector(xb);
  batched.initialize_dof_vector(yb);
  {
    HostVector x_layout_b(batched.get_domain_partitioner());
    x_layout_b.copy_locally_owned_data_from(x);
    xb.import_elements(x_layout_b, VectorOperation::insert);
  }
  batched.vmult(yb, xb);
  pcout << "vmult vs host matrix-free: "
        << verdict(relative_difference(y_mf, yb)) << std::endl;
  pcout << "vmult vs assembled matrix: "
        << verdict(relative_difference(y_mat, yb)) << std::endl;
  {
    DeviceVector x_plain(dof_handler.locally_owned_dofs(), comm);
    x_plain.import_elements(x_owned, VectorOperation::insert);
    DeviceVector y2(dof_handler.locally_owned_dofs(), comm);
    batched.vmult(y2, x_plain);
    pcout << "vmult with plain source and destination: "
          << verdict(relative_difference(y_mat, y2)) << std::endl;

    HostVector y_twice(y_mat);
    y_twice *= 2.;
    batched.vmult_add(y2, x_plain);
    pcout << "vmult_add: " << verdict(relative_difference(y_twice, y2))
          << std::endl;

    batched.Tvmult(y2, x_plain);
    pcout << "Tvmult: " << verdict(relative_difference(y_mat, y2)) << std::endl;

    // ghosted source: the operator must not disturb a vector whose ghosts
    // are already up to date
    xb.update_ghost_values();
    batched.vmult(yb, xb);
    pcout << "vmult with ghosted source: "
          << verdict(relative_difference(y_mat, yb)) << std::endl;
    xb.zero_out_ghost_values();
  }
  {
    DeviceVector diag_dev(dof_handler.locally_owned_dofs(), comm);
    batched.get_diagonal(diag_dev);
    pcout << "diagonal: " << verdict(relative_difference(diag, diag_dev))
          << std::endl;
    batched.get_inverse_diagonal(diag_dev);
    pcout << "inverse diagonal: "
          << verdict(relative_difference(inv_diag, diag_dev)) << std::endl;
    DeviceVector diag_layout;
    batched.initialize_dof_vector(diag_layout);
    batched.get_diagonal(diag_layout);
    pcout << "diagonal in operator layout: "
          << verdict(relative_difference(diag, diag_layout)) << std::endl;
  }

  // ---------------------------------------------------------------------
  // Batched operator, original cell_extent penalty
  // ---------------------------------------------------------------------
  pcout << "-- PortableDG::LaplaceOperatorDG, cell_extent penalty --"
        << std::endl;
  PortableDG::LaplaceOperatorDG<dim, degree> batched_extent;
  {
    typename PortableDG::LaplaceOperatorDG<dim, degree>::AdditionalData data;
    data.penalty_convention = PortableDG::PenaltyConvention::cell_extent;
    batched_extent.reinit(mapping, dof_handler, data);
  }
  {
    // symmetry: <A x, z> == <x, A z>
    HostVector z(x);
    fill_test_vector(z, 0.7);
    HostVector z_layout(batched_extent.get_domain_partitioner());
    z_layout.copy_locally_owned_data_from(z);
    DeviceVector zb, Ax, Az;
    batched_extent.initialize_dof_vector(zb);
    batched_extent.initialize_dof_vector(Ax);
    batched_extent.initialize_dof_vector(Az);
    zb.import_elements(z_layout, VectorOperation::insert);
    batched_extent.vmult(Ax, xb);
    batched_extent.vmult(Az, zb);
    const double axz = Ax * zb;
    const double xaz = xb * Az;
    pcout << "symmetry: "
          << verdict(std::abs(axz - xaz) / std::max(std::abs(axz), 1e-300))
          << std::endl;
    const double xax = xb * Ax;
    pcout << "positivity: " << (xax > 0 ? "OK" : "FAIL") << std::endl;

    // On Cartesian cells the two conventions differ only in the penalty
    // weight (factor two), so A_59 - A_ext equals A_ext - A_0 with A_0 the
    // operator without penalty: check (A_59 - A_ext) x against the exact
    // penalty-only contribution computed on the host.
    if (!distort)
      {
        DeviceVector A59x;
        batched.initialize_dof_vector(A59x);
        batched.vmult(A59x, xb);
        A59x.add(-1., Ax); // = (A_59 - A_ext) x = S x

        // S x on the host: sum over faces of sigma_ext [x] [phi_i], Dirichlet
        // faces 2 sigma_b x phi_i, with sigma_ext = p(p+1)/2 (1/h_- + 1/h_+)
        HostVector x_ghosted(dof_handler.locally_owned_dofs(),
                             DoFTools::extract_locally_relevant_dofs(
                               dof_handler),
                             comm);
        x_ghosted.copy_locally_owned_data_from(x);
        x_ghosted.update_ghost_values();
        HostVector         Sx(dof_handler.locally_owned_dofs(), comm);
        QGauss<dim - 1>    face_quadrature(degree + 1);
        FEFaceValues<dim>  fe_face(mapping,
                                  fe_dg,
                                  face_quadrature,
                                  update_values | update_JxW_values);
        FEFaceValues<dim>  fe_face_n(mapping,
                                    fe_dg,
                                    face_quadrature,
                                    update_values);
        const unsigned int n_dofs = fe_dg.n_dofs_per_cell();
        std::vector<types::global_dof_index> dofs(n_dofs), dofs_n(n_dofs);
        const double                         factor = degree * (degree + 1.);
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(dofs);
              for (const unsigned int f : cell->face_indices())
                {
                  fe_face.reinit(cell, f);
                  const double extent_m =
                    cell->measure() / cell->face(f)->measure();
                  if (cell->at_boundary(f))
                    {
                      const double sigma_b = factor / extent_m;
                      for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                        {
                          double u = 0;
                          for (unsigned int j = 0; j < n_dofs; ++j)
                            u += x_ghosted(dofs[j]) * fe_face.shape_value(j, q);
                          for (unsigned int i = 0; i < n_dofs; ++i)
                            Sx(dofs[i]) += 2. * sigma_b * u *
                                           fe_face.shape_value(i, q) *
                                           fe_face.JxW(q);
                        }
                    }
                  else
                    {
                      const auto         neighbor = cell->neighbor(f);
                      const unsigned int nf = cell->neighbor_of_neighbor(f);
                      neighbor->get_dof_indices(dofs_n);
                      fe_face_n.reinit(neighbor, nf);
                      const double extent_p =
                        neighbor->measure() / neighbor->face(nf)->measure();
                      const double sigma =
                        factor * 0.5 * (1. / extent_m + 1. / extent_p);
                      for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                        {
                          double jump = 0;
                          for (unsigned int j = 0; j < n_dofs; ++j)
                            jump +=
                              x_ghosted(dofs[j]) * fe_face.shape_value(j, q) -
                              x_ghosted(dofs_n[j]) *
                                fe_face_n.shape_value(j, q);
                          for (unsigned int i = 0; i < n_dofs; ++i)
                            Sx(dofs[i]) += sigma * jump *
                                           fe_face.shape_value(i, q) *
                                           fe_face.JxW(q);
                        }
                    }
                }
            }
        pcout << "A_59 - A_ext equals the penalty difference: "
              << verdict(relative_difference(Sx, A59x)) << std::endl;
      }
  }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm                   comm = MPI_COMM_WORLD;

  run<2, 1>(comm, 4, false);
  run<2, 2>(comm, 4, false);
  run<2, 2>(comm, 4, true);
  run<3, 1>(comm, 2, false);
  run<3, 2>(comm, 2, false);
  run<3, 2>(comm, 2, true);
}
