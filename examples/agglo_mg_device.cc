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


// Agglomerated multigrid on the device.
//
// The setup (R-tree agglomeration, injection matrices, Galerkin projections)
// runs on the host with Trilinos. Every level matrix and transfer matrix is
// then copied once into a DeviceCSRMatrix, and the complete V-cycle
// (Chebyshev smoother, transfers, outer CG) operates on
// LinearAlgebra::distributed::Vector<double, MemorySpace::Default>. Only the
// coarsest level is solved on the host by a direct solver. The device solve is
// run three times: with the assembled fine matrix, with the matrix-free
// DeviceDGOperator, and with the batched sum-factorised
// PortableDG::LaplaceOperatorDG on the finest level.
//
// With a deal.II built on a Serial or OpenMP Kokkos backend the "device" is
// the host, and this example then measures the CSR kernels against Trilinos.
// With a CUDA or HIP backend the same binary runs the solve on the GPU.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/multigrid.h>

#include <agglomeration_handler.h>
#include <agglomerator.h>
#include <device_dg_operator.h>
#include <device_sparse_matrix.h>
#include <multigrid_amg.h>
#include <poly_utils.h>
#include <portable/laplace_operator_dg.h>
#include <utils.h>

using namespace dealii;

static constexpr unsigned int dim                   = 3;
static constexpr unsigned int degree_finite_element = 2;
static constexpr unsigned int n_global_refinements  = 4;
static constexpr unsigned int starting_tree_level   = 1;

using HostVector = LinearAlgebra::distributed::Vector<double>;
using DeviceVector =
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm                   comm = MPI_COMM_WORLD;
  const unsigned int n_ranks            = Utilities::MPI::n_mpi_processes(comm);
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(comm) == 0);

  pcout << "Running with " << n_ranks << " MPI ranks, degree "
        << degree_finite_element << ", Kokkos execution space: "
        << typeid(MemorySpace::Default::kokkos_space::execution_space).name()
        << std::endl;

  // ---------------------------------------------------------------------
  // Fine grid and fine operator
  // ---------------------------------------------------------------------
  Triangulation<dim> serial_tria;
  GridGenerator::hyper_cube(serial_tria, 0., 1.);
  serial_tria.refine_global(n_global_refinements);
  GridTools::partition_triangulation(n_ranks, serial_tria);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial_tria, comm);
  parallel::fullydistributed::Triangulation<dim> tria(comm);
  tria.create_triangulation(description);
  pcout << "Fine cells: " << tria.n_global_active_cells() << std::endl;

  FE_DGQ<dim>     fe_dg(degree_finite_element);
  MappingQ1<dim>  mapping;
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_dg);

  Utils::MatrixFreeOperators::
    LaplaceOperatorDG<dim, degree_finite_element, degree_finite_element + 1, 1>
      fine_operator;
  fine_operator.reinit(mapping, dof_handler);

  // ---------------------------------------------------------------------
  // Agglomerated levels, transfers and Galerkin projections (host)
  // ---------------------------------------------------------------------
  Timer                                                   setup_timer;
  std::vector<std::unique_ptr<AgglomerationHandler<dim>>> handlers;
  const unsigned int max_level = PolyUtils::construct_agglomerated_levels(
    tria, handlers, fe_dg, mapping, starting_tree_level);

  MGLevelObject<TrilinosWrappers::SparseMatrix> transfers(0, max_level - 1);
  for (unsigned int l = 0; l < max_level; ++l)
    {
      SparsityPattern sparsity;
      if (l + 1 < max_level)
        Utils::fill_injection_matrix(*handlers[l],
                                     *handlers[l + 1],
                                     sparsity,
                                     transfers[l]);
      else
        PolyUtils::fill_interpolation_matrix(*handlers.back(), transfers[l]);
    }

  MGLevelObject<std::unique_ptr<TrilinosWrappers::SparseMatrix>> level_matrices(
    0, max_level);
  level_matrices[max_level] =
    std::make_unique<TrilinosWrappers::SparseMatrix>();
  fine_operator.get_system_matrix(*level_matrices[max_level]);
  AmgProjector<dim, TrilinosWrappers::SparseMatrix> projector(transfers);
  projector.compute_level_matrices(level_matrices);

  std::vector<DoFHandler<dim> *> dof_handlers(max_level + 1);
  for (unsigned int l = 0; l < max_level; ++l)
    dof_handlers[l] = &handlers[l]->agglo_dh;
  dof_handlers[max_level] = &dof_handler;

  for (unsigned int l = 0; l <= max_level; ++l)
    pcout << "Level " << l << ": " << dof_handlers[l]->n_dofs() << " DoFs, "
          << level_matrices[l]->n_nonzero_elements() << " nonzeros"
          << std::endl;
  setup_timer.stop();
  pcout << "Host setup (agglomeration + projections): "
        << setup_timer.wall_time() << " s" << std::endl;

  HostVector rhs;
  fine_operator.initialize_dof_vector(rhs);
  fine_operator.rhs(rhs);

  ReductionControl control(1000, 1e-13, 1e-9);

  // ---------------------------------------------------------------------
  // Reference: the same V-cycle with Trilinos matrices and host vectors
  // ---------------------------------------------------------------------
  {
    using LevelMatrixType = LinearOperatorMG<HostVector, HostVector>;
    MGLevelObject<LevelMatrixType> level_ops(0, max_level);
    for (unsigned int l = 0; l <= max_level; ++l)
      {
        level_ops[l] =
          linear_operator_mg<HostVector, HostVector>(*level_matrices[l]);
        level_ops[l].n_rows = level_matrices[l]->m();
        level_ops[l].n_cols = level_matrices[l]->n();
      }
    mg::Matrix<HostVector> mg_matrix(level_ops);

    using SmootherType = PreconditionChebyshev<LevelMatrixType, HostVector>;
    mg::SmootherRelaxation<SmootherType, HostVector>     mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data(
      0, max_level);
    std::vector<HostVector> inverse_diagonals(max_level + 1);
    for (unsigned int l = 0; l <= max_level; ++l)
      {
        inverse_diagonals[l].reinit(dof_handlers[l]->locally_owned_dofs(),
                                    comm);
        for (const auto i : dof_handlers[l]->locally_owned_dofs())
          inverse_diagonals[l](i) = 1. / level_matrices[l]->diag_element(i);
        smoother_data[l].preconditioner =
          std::make_shared<DiagonalMatrix<HostVector>>(inverse_diagonals[l]);
        smoother_data[l].smoothing_range     = 20.;
        smoother_data[l].degree              = 3;
        smoother_data[l].eig_cg_n_iterations = 20;
      }
    mg_smoother.set_steps(2);
    mg_smoother.initialize(level_ops, smoother_data);

    Utils::MGCoarseDirect<HostVector,
                          TrilinosWrappers::SparseMatrix,
                          TrilinosWrappers::SolverDirect>
      mg_coarse(*level_matrices[0]);

    MGTransferAgglomeration<dim, HostVector> mg_transfer(transfers,
                                                         dof_handlers);
    Multigrid<HostVector>                    mg(
      mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    PreconditionMG<dim, HostVector, MGTransferAgglomeration<dim, HostVector>>
      preconditioner(dof_handler, mg, mg_transfer);

    HostVector solution;
    fine_operator.initialize_dof_vector(solution);
    SolverCG<HostVector> cg(control);
    Timer                solve_timer;
    cg.solve(*level_matrices[max_level], solution, rhs, preconditioner);
    solve_timer.stop();
    pcout << "Host   CG+MG: " << control.last_step() << " iterations, "
          << solve_timer.wall_time() << " s" << std::endl;
  }

  // ---------------------------------------------------------------------
  // Device V-cycle
  // ---------------------------------------------------------------------
  {
    using LevelMatrixType = DeviceCSRMatrix<double>;

    Timer                          copy_timer;
    MGLevelObject<LevelMatrixType> level_dev(0, max_level);
    for (unsigned int l = 0; l <= max_level; ++l)
      level_dev[l].reinit(*level_matrices[l]);
    MGLevelObject<LevelMatrixType> transfers_dev(0, max_level - 1);
    for (unsigned int l = 0; l < max_level; ++l)
      transfers_dev[l].reinit(transfers[l]);
    copy_timer.stop();

    std::size_t device_bytes = 0;
    for (unsigned int l = 0; l <= max_level; ++l)
      device_bytes += level_dev[l].memory_consumption();
    for (unsigned int l = 0; l < max_level; ++l)
      device_bytes += transfers_dev[l].memory_consumption();
    device_bytes = Utilities::MPI::sum(device_bytes, comm);
    pcout << "Copied matrices to the device in " << copy_timer.wall_time()
          << " s, device storage " << device_bytes / (1024. * 1024.) << " MiB"
          << std::endl;

    mg::Matrix<DeviceVector> mg_matrix(level_dev);

    using SmootherType = PreconditionChebyshev<LevelMatrixType, DeviceVector>;
    mg::SmootherRelaxation<SmootherType, DeviceVector>   mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data(
      0, max_level);
    std::vector<DeviceVector> inverse_diagonals(max_level + 1);
    for (unsigned int l = 0; l <= max_level; ++l)
      {
        inverse_diagonals[l].reinit(dof_handlers[l]->locally_owned_dofs(),
                                    comm);
        level_dev[l].get_inverse_diagonal(inverse_diagonals[l]);
        smoother_data[l].preconditioner =
          std::make_shared<DiagonalMatrix<DeviceVector>>(inverse_diagonals[l]);
        smoother_data[l].smoothing_range     = 20.;
        smoother_data[l].degree              = 3;
        smoother_data[l].eig_cg_n_iterations = 20;
      }
    mg_smoother.set_steps(2);
    mg_smoother.initialize(level_dev, smoother_data);

    Utils::MGCoarseDirect<DeviceVector,
                          TrilinosWrappers::SparseMatrix,
                          TrilinosWrappers::SolverDirect>
      mg_coarse(*level_matrices[0]);

    using TransferType =
      MGTransferAgglomeration<dim, DeviceVector, LevelMatrixType>;
    TransferType            mg_transfer(transfers_dev, dof_handlers);
    Multigrid<DeviceVector> mg(
      mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    PreconditionMG<dim, DeviceVector, TransferType> preconditioner(dof_handler,
                                                                   mg,
                                                                   mg_transfer);

    HostVector rhs_layout(level_dev[max_level].get_domain_partitioner());
    rhs_layout.copy_locally_owned_data_from(rhs);
    DeviceVector rhs_dev, solution_dev;
    level_dev[max_level].initialize_dof_vector(rhs_dev);
    level_dev[max_level].initialize_dof_vector(solution_dev);
    rhs_dev.import_elements(rhs_layout, VectorOperation::insert);

    SolverCG<DeviceVector> cg(control);
    Timer                  solve_timer;
    cg.solve(level_dev[max_level], solution_dev, rhs_dev, preconditioner);
    solve_timer.stop();
    pcout << "Device CG+MG (assembled fine level): " << control.last_step()
          << " iterations, " << solve_timer.wall_time() << " s" << std::endl;

    // -------------------------------------------------------------------
    // Matrix-free finest level
    // -------------------------------------------------------------------
    Timer                                        mf_setup_timer;
    DeviceDGOperator<dim, degree_finite_element> fine_mf;
    fine_mf.reinit(mapping, dof_handler);
    mf_setup_timer.stop();
    const std::size_t mf_bytes =
      Utilities::MPI::sum(fine_mf.memory_consumption(), comm);
    const std::size_t fine_matrix_bytes =
      Utilities::MPI::sum(level_dev[max_level].memory_consumption(), comm);
    pcout << "Matrix-free fine operator set up in "
          << mf_setup_timer.wall_time() << " s, device storage "
          << mf_bytes / (1024. * 1024.) << " MiB (assembled fine matrix: "
          << fine_matrix_bytes / (1024. * 1024.) << " MiB)" << std::endl;

    using LevelOperator = LinearOperatorMG<DeviceVector, DeviceVector>;
    MGLevelObject<LevelOperator> level_ops(0, max_level);
    for (unsigned int l = 0; l < max_level; ++l)
      {
        level_ops[l] =
          linear_operator_mg<DeviceVector, DeviceVector>(level_dev[l]);
        level_ops[l].n_rows = level_dev[l].m();
        level_ops[l].n_cols = level_dev[l].n();
      }
    level_ops[max_level] =
      linear_operator_mg<DeviceVector, DeviceVector>(fine_mf);
    level_ops[max_level].n_rows = fine_mf.m();
    level_ops[max_level].n_cols = fine_mf.n();
    mg::Matrix<DeviceVector> mg_matrix_mf(level_ops);

    using SmootherMF = PreconditionChebyshev<LevelOperator, DeviceVector>;
    mg::SmootherRelaxation<SmootherMF, DeviceVector>   mg_smoother_mf;
    MGLevelObject<typename SmootherMF::AdditionalData> smoother_data_mf(
      0, max_level);
    DeviceVector fine_inverse_diagonal(dof_handler.locally_owned_dofs(), comm);
    fine_mf.get_inverse_diagonal(fine_inverse_diagonal);
    for (unsigned int l = 0; l <= max_level; ++l)
      {
        smoother_data_mf[l].preconditioner =
          std::make_shared<DiagonalMatrix<DeviceVector>>(
            l < max_level ? inverse_diagonals[l] : fine_inverse_diagonal);
        smoother_data_mf[l].smoothing_range     = 20.;
        smoother_data_mf[l].degree              = 3;
        smoother_data_mf[l].eig_cg_n_iterations = 20;
      }
    mg_smoother_mf.set_steps(2);
    mg_smoother_mf.initialize(level_ops, smoother_data_mf);

    Multigrid<DeviceVector> mg_mf(
      mg_matrix_mf, mg_coarse, mg_transfer, mg_smoother_mf, mg_smoother_mf);
    PreconditionMG<dim, DeviceVector, TransferType> preconditioner_mf(
      dof_handler, mg_mf, mg_transfer);

    HostVector rhs_layout_mf(fine_mf.get_domain_partitioner());
    rhs_layout_mf.copy_locally_owned_data_from(rhs);
    DeviceVector rhs_mf, solution_mf;
    fine_mf.initialize_dof_vector(rhs_mf);
    fine_mf.initialize_dof_vector(solution_mf);
    rhs_mf.import_elements(rhs_layout_mf, VectorOperation::insert);

    SolverCG<DeviceVector> cg_mf(control);
    Timer                  mf_solve_timer;
    cg_mf.solve(fine_mf, solution_mf, rhs_mf, preconditioner_mf);
    mf_solve_timer.stop();
    pcout << "Device CG+MG (matrix-free fine level): " << control.last_step()
          << " iterations, " << mf_solve_timer.wall_time() << " s" << std::endl;

    // -------------------------------------------------------------------
    // Batched sum-factorised finest level
    // -------------------------------------------------------------------
    Timer batched_setup_timer;
    PortableDG::LaplaceOperatorDG<dim, degree_finite_element> fine_batched;
    fine_batched.reinit(mapping, dof_handler);
    batched_setup_timer.stop();
    const std::size_t batched_bytes =
      Utilities::MPI::sum(fine_batched.memory_consumption(), comm);
    pcout << "Batched fine operator set up in "
          << batched_setup_timer.wall_time() << " s, device storage "
          << batched_bytes / (1024. * 1024.) << " MiB" << std::endl;

    level_ops[max_level] =
      linear_operator_mg<DeviceVector, DeviceVector>(fine_batched);
    level_ops[max_level].n_rows = fine_batched.m();
    level_ops[max_level].n_cols = fine_batched.n();
    mg::Matrix<DeviceVector> mg_matrix_batched(level_ops);

    mg::SmootherRelaxation<SmootherMF, DeviceVector> mg_smoother_batched;
    fine_batched.get_inverse_diagonal(fine_inverse_diagonal);
    smoother_data_mf[max_level].preconditioner =
      std::make_shared<DiagonalMatrix<DeviceVector>>(fine_inverse_diagonal);
    mg_smoother_batched.set_steps(2);
    mg_smoother_batched.initialize(level_ops, smoother_data_mf);

    Multigrid<DeviceVector> mg_batched(mg_matrix_batched,
                                       mg_coarse,
                                       mg_transfer,
                                       mg_smoother_batched,
                                       mg_smoother_batched);
    PreconditionMG<dim, DeviceVector, TransferType> preconditioner_batched(
      dof_handler, mg_batched, mg_transfer);

    HostVector rhs_layout_b(fine_batched.get_domain_partitioner());
    rhs_layout_b.copy_locally_owned_data_from(rhs);
    DeviceVector rhs_b, solution_b;
    fine_batched.initialize_dof_vector(rhs_b);
    fine_batched.initialize_dof_vector(solution_b);
    rhs_b.import_elements(rhs_layout_b, VectorOperation::insert);

    SolverCG<DeviceVector> cg_b(control);
    Timer                  batched_solve_timer;
    cg_b.solve(fine_batched, solution_b, rhs_b, preconditioner_batched);
    batched_solve_timer.stop();
    pcout << "Device CG+MG (batched matrix-free fine level): "
          << control.last_step() << " iterations, "
          << batched_solve_timer.wall_time() << " s" << std::endl;
  }
}
