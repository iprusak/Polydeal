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


// Check DeviceCSRMatrix against TrilinosWrappers::SparseMatrix on matrices
// produced by polyDEAL itself: the injection matrix between two agglomerated
// levels (rectangular) and the SIP-DG matrix on the finer polytopal level
// (square). Products, transposed products, accumulating products and the
// diagonal are compared. A serial deal.II SparseMatrix is checked as well.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <agglomeration_handler.h>
#include <agglomerator.h>
#include <device_sparse_matrix.h>
#include <poly_utils.h>
#include <utils.h>

#include <cmath>

using namespace dealii;

using HostVector = LinearAlgebra::distributed::Vector<double>;
using DeviceVector =
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;



// Deterministic, rank-count independent entries.
void
fill_test_vector(HostVector &v, const double shift)
{
  for (const types::global_dof_index i : v.locally_owned_elements())
    v(i) = std::sin(0.37 * i + shift) + 0.1 * (i % 7);
}



double
relative_difference(const HostVector &reference, const DeviceVector &device)
{
  HostVector back;
  back.reinit(reference);
  back.import_elements(device, VectorOperation::insert);
  back -= reference;
  const double ref_norm = reference.l2_norm();
  return back.l2_norm() / (ref_norm > 0 ? ref_norm : 1.);
}



std::string
verdict(const double rel)
{
  return rel < 1e-12 ? "OK" : "FAIL (" + std::to_string(rel) + ")";
}



template <int dim>
void
agglomerate_level(
  AgglomerationHandler<dim> &ah,
  const Triangulation<dim>  &tria,
  const std::vector<
    std::vector<typename Triangulation<dim>::active_cell_iterator>>
                           &agglomerates,
  const FiniteElement<dim> &fe)
{
  // Same procedure as in distributed_injection_01.cc: flag cells by material
  // id, then collect the locally owned ones per agglomerate.
  std::size_t agglo_index = 0;
  for (const auto &agglo : agglomerates)
    {
      for (const auto &el : agglo)
        el->set_material_id(agglo_index);
      ++agglo_index;
    }

  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
    cells_per_id(agglo_index);
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      cells_per_id[cell->material_id()].push_back(cell);

  for (const auto &cells : cells_per_id)
    ah.define_agglomerate(cells);

  ah.initialize_fe_values(QGauss<dim>(fe.degree + 1),
                          update_values | update_gradients | update_JxW_values |
                            update_quadrature_points,
                          QGauss<dim - 1>(fe.degree + 1),
                          update_JxW_values);
  ah.distribute_agglomerated_dofs(fe);
}



template <int dim>
void
run_distributed(const unsigned int degree, const MPI_Comm comm)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(comm) == 0);
  const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(comm);
  pcout << "*** Distributed check, dim=" << dim << ", degree=" << degree
        << ", ranks=" << n_ranks << " ***" << std::endl;

  // Fine grid
  Triangulation<dim> serial_tria;
  GridGenerator::hyper_cube(serial_tria, 0., 1.);
  serial_tria.refine_global(5);
  GridTools::partition_triangulation(n_ranks, serial_tria);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial_tria, comm);
  parallel::fullydistributed::Triangulation<dim> tria(comm);
  tria.create_triangulation(description);

  GridTools::Cache<dim> cached_tria(tria);
  FE_DGQ<dim>           fe_dg(degree);
  MappingQ1<dim>        mapping;

  // R-tree of the locally owned cells
  namespace bgi = boost::geometry::index;
  static constexpr unsigned int max_elem_per_node =
    Utils::constexpr_pow(2, dim);
  std::vector<std::pair<BoundingBox<dim>,
                        typename Triangulation<dim>::active_cell_iterator>>
               boxes(tria.n_locally_owned_active_cells());
  unsigned int i = 0;
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);
  auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(boxes);

  const unsigned int extraction_level = 2;

  AgglomerationHandler<dim>              ah_coarse(cached_tria);
  CellsAgglomerator<dim, decltype(tree)> agglomerator_coarse{tree,
                                                             extraction_level};
  const auto agglomerates_coarse = agglomerator_coarse.extract_agglomerates();
  ah_coarse.connect_hierarchy(agglomerator_coarse);
  agglomerate_level(ah_coarse, tria, agglomerates_coarse, fe_dg);

  AgglomerationHandler<dim>              ah_fine(cached_tria);
  CellsAgglomerator<dim, decltype(tree)> agglomerator_fine{tree,
                                                           extraction_level +
                                                             1};
  const auto agglomerates_fine = agglomerator_fine.extract_agglomerates();
  agglomerate_level(ah_fine, tria, agglomerates_fine, fe_dg);

  pcout << "Coarse DoFs: " << ah_coarse.n_dofs()
        << ", fine DoFs: " << ah_fine.n_dofs() << std::endl;

  // ----- Rectangular matrix: injection from coarse to fine -----
  SparsityPattern                embedding_sp;
  TrilinosWrappers::SparseMatrix injection;
  Utils::fill_injection_matrix(ah_coarse, ah_fine, embedding_sp, injection);

  DeviceCSRMatrix<double> injection_dev(injection);
  pcout << "Injection matrix: " << injection.m() << " x " << injection.n()
        << ", nnz " << injection.n_nonzero_elements() << " (device nnz "
        << injection_dev.n_nonzero_elements() << ")" << std::endl;
  AssertThrow(injection_dev.m() == injection.m() &&
                injection_dev.n() == injection.n() &&
                injection_dev.n_nonzero_elements() ==
                  injection.n_nonzero_elements(),
              ExcInternalError());

  const IndexSet &owned_coarse = ah_coarse.agglo_dh.locally_owned_dofs();
  const IndexSet &owned_fine   = ah_fine.agglo_dh.locally_owned_dofs();

  HostVector x_coarse(owned_coarse, comm), x_fine(owned_fine, comm);
  HostVector y_fine(owned_fine, comm), y_coarse(owned_coarse, comm);
  fill_test_vector(x_coarse, 0.1);
  fill_test_vector(x_fine, 0.2);

  DeviceVector x_coarse_dev, x_fine_dev, y_fine_dev, y_coarse_dev;
  x_coarse_dev.reinit(x_coarse.get_partitioner());
  x_fine_dev.reinit(x_fine.get_partitioner());
  x_coarse_dev.import_elements(x_coarse, VectorOperation::insert);
  x_fine_dev.import_elements(x_fine, VectorOperation::insert);

  // vmult with a non-ghosted source (forces the internal copy path)
  injection.vmult(y_fine, x_coarse);
  y_fine_dev.reinit(y_fine.get_partitioner());
  injection_dev.vmult(y_fine_dev, x_coarse_dev);
  pcout << "Injection vmult (plain source): "
        << verdict(relative_difference(y_fine, y_fine_dev)) << std::endl;

  // vmult with a source in the matrix's own domain layout (no copy)
  {
    DeviceVector x_layout;
    injection_dev.initialize_domain_vector(x_layout);
    x_layout.copy_locally_owned_data_from(x_coarse_dev);
    DeviceVector y_layout;
    injection_dev.initialize_range_vector(y_layout);
    injection_dev.vmult(y_layout, x_layout);
    pcout << "Injection vmult (domain layout): "
          << verdict(relative_difference(y_fine, y_layout)) << std::endl;
  }

  // vmult_add
  {
    HostVector y2(y_fine);
    injection.vmult_add(y2, x_coarse);
    injection_dev.vmult_add(y_fine_dev, x_coarse_dev);
    pcout << "Injection vmult_add: "
          << verdict(relative_difference(y2, y_fine_dev)) << std::endl;
  }

  // Tvmult
  injection.Tvmult(y_coarse, x_fine);
  y_coarse_dev.reinit(y_coarse.get_partitioner());
  injection_dev.Tvmult(y_coarse_dev, x_fine_dev);
  pcout << "Injection Tvmult: "
        << verdict(relative_difference(y_coarse, y_coarse_dev)) << std::endl;

  // Tvmult_add
  {
    HostVector y2(y_coarse);
    injection.Tvmult_add(y2, x_fine);
    injection_dev.Tvmult_add(y_coarse_dev, x_fine_dev);
    pcout << "Injection Tvmult_add: "
          << verdict(relative_difference(y2, y_coarse_dev)) << std::endl;
  }

  // ----- Square matrix: SIP-DG on the fine polytopal level -----
  TrilinosWrappers::SparseMatrix system_matrix;
  PolyUtils::assemble_dg_matrix(system_matrix, fe_dg, ah_fine);

  DeviceCSRMatrix<double> system_dev(system_matrix);
  pcout << "SIP matrix: " << system_matrix.m() << " x " << system_matrix.n()
        << ", nnz " << system_matrix.n_nonzero_elements() << " (device nnz "
        << system_dev.n_nonzero_elements() << ")" << std::endl;

  HostVector   z_fine(owned_fine, comm);
  DeviceVector z_fine_dev;
  z_fine_dev.reinit(z_fine.get_partitioner());

  system_matrix.vmult(z_fine, x_fine);
  system_dev.vmult(z_fine_dev, x_fine_dev);
  pcout << "SIP vmult: " << verdict(relative_difference(z_fine, z_fine_dev))
        << std::endl;

  system_matrix.Tvmult(z_fine, x_fine);
  system_dev.Tvmult(z_fine_dev, x_fine_dev);
  pcout << "SIP Tvmult: " << verdict(relative_difference(z_fine, z_fine_dev))
        << std::endl;

  // Diagonal
  {
    HostVector diag(owned_fine, comm);
    for (const types::global_dof_index i : owned_fine)
      diag(i) = system_matrix.diag_element(i);

    DeviceVector diag_dev;
    diag_dev.reinit(z_fine.get_partitioner());
    system_dev.get_diagonal(diag_dev);
    pcout << "SIP diagonal: " << verdict(relative_difference(diag, diag_dev))
          << std::endl;

    HostVector inv_diag(diag);
    for (const types::global_dof_index i : owned_fine)
      inv_diag(i) = 1. / diag(i);
    system_dev.get_inverse_diagonal(diag_dev);
    pcout << "SIP inverse diagonal: "
          << verdict(relative_difference(inv_diag, diag_dev)) << std::endl;
  }

  // Memory report, independent of the number of ranks only in the sum.
  const std::size_t bytes =
    Utilities::MPI::sum(system_dev.memory_consumption(), comm);
  pcout << "Device storage for the SIP matrix (forward + transpose): " << bytes
        << " bytes" << std::endl;
}



void
run_serial(const MPI_Comm comm)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(comm) == 0);
  pcout << "*** Serial check ***" << std::endl;

  // Non-symmetric, rectangular 1D finite-difference-like matrix.
  const unsigned int     n_rows = 23, n_cols = 29;
  DynamicSparsityPattern dsp(n_rows, n_cols);
  for (unsigned int r = 0; r < n_rows; ++r)
    {
      dsp.add(r, r);
      dsp.add(r, r + 1);
      dsp.add(r, (r + 5) % n_cols);
      dsp.add(r, n_cols - 1 - r);
    }
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A(sp);
  for (unsigned int r = 0; r < n_rows; ++r)
    {
      A.set(r, r, 2. + r);
      A.set(r, r + 1, -1. + 0.01 * r);
      A.add(r, (r + 5) % n_cols, 0.5);
      A.add(r, n_cols - 1 - r, -0.25 * r);
    }

  DeviceCSRMatrix<double> A_dev(A);

  HostVector x(n_cols), y(n_rows), xt(n_rows), yt(n_cols);
  fill_test_vector(x, 0.3);
  fill_test_vector(xt, 0.4);
  A.vmult(y, x);
  A.Tvmult(yt, xt);

  DeviceVector x_dev, y_dev, xt_dev, yt_dev;
  x_dev.reinit(x.get_partitioner());
  xt_dev.reinit(xt.get_partitioner());
  y_dev.reinit(y.get_partitioner());
  yt_dev.reinit(yt.get_partitioner());
  x_dev.import_elements(x, VectorOperation::insert);
  xt_dev.import_elements(xt, VectorOperation::insert);

  A_dev.vmult(y_dev, x_dev);
  A_dev.Tvmult(yt_dev, xt_dev);
  pcout << "Serial vmult: " << verdict(relative_difference(y, y_dev))
        << std::endl;
  pcout << "Serial Tvmult: " << verdict(relative_difference(yt, yt_dev))
        << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm                   comm = MPI_COMM_WORLD;

  run_serial(comm);
  run_distributed<2>(1, comm);
  run_distributed<2>(2, comm);
}
