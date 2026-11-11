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


// Agglomerated multigrid with every level, transfer and smoother operating on
// LinearAlgebra::distributed::Vector<double, MemorySpace::Default>. The level
// matrices are Galerkin projections computed on the host with Trilinos and
// copied once into DeviceCSRMatrix objects. The same V-cycle is run with host
// vectors and Trilinos matrices, and the two CG solves are compared. A third
// run replaces the assembled fine matrix by the matrix-free DeviceDGOperator,
// a fourth by the batched PortableDG::LaplaceOperatorDG.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
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

#include <iomanip>

using namespace dealii;

using HostVector = LinearAlgebra::distributed::Vector<double>;
using DeviceVector =
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;



template <int dim, int degree>
void
run(const MPI_Comm comm)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(comm) == 0);
  const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(comm);
  pcout << "*** dim=" << dim << ", degree=" << degree << ", ranks=" << n_ranks
        << " ***" << std::endl;

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

  FE_DGQ<dim>     fe_dg(degree);
  MappingQ1<dim>  mapping;
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_dg);

  // Fine operator (matrix-free on the host, used here only to assemble the
  // fine matrix and the right-hand side).
  Utils::MatrixFreeOperators::LaplaceOperatorDG<dim, degree, degree + 1, 1>
    fine_operator;
  fine_operator.reinit(mapping, dof_handler);

  // Agglomerated levels from the R-tree
  std::vector<std::unique_ptr<AgglomerationHandler<dim>>> handlers;
  const unsigned int                                      starting_level = 1;
  const unsigned int n_agglo_levels = PolyUtils::construct_agglomerated_levels(
    tria, handlers, fe_dg, mapping, starting_level);
  const unsigned int max_level = n_agglo_levels; // finest = classical DoFs

  // Two-level transfers P_l: level l -> level l+1
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

  // Level matrices by Galerkin projection, on the host
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

  // Right-hand side (source term 1) on the host
  HostVector rhs;
  fine_operator.initialize_dof_vector(rhs);
  fine_operator.rhs(rhs);

  ReductionControl control(1000, 1e-13, 1e-9);

  // ---------------------------------------------------------------------
  // Host pipeline: Trilinos matrices, host vectors
  // ---------------------------------------------------------------------
  HostVector   solution_host;
  unsigned int host_iterations = 0;
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

    fine_operator.initialize_dof_vector(solution_host);
    SolverCG<HostVector> cg(control);
    cg.solve(*level_matrices[max_level], solution_host, rhs, preconditioner);
    host_iterations = control.last_step();
    pcout << "Host   CG+MG: " << host_iterations << " iterations, residual "
          << std::scientific << std::setprecision(2) << control.last_value()
          << std::defaultfloat << std::endl;
  }

  // ---------------------------------------------------------------------
  // Device pipeline: DeviceCSRMatrix on every level, device vectors
  // ---------------------------------------------------------------------
  DeviceVector solution_dev, solution_mf, solution_batched;
  {
    using LevelMatrixType = DeviceCSRMatrix<double>;
    MGLevelObject<LevelMatrixType> level_dev(0, max_level);
    for (unsigned int l = 0; l <= max_level; ++l)
      level_dev[l].reinit(*level_matrices[l]);

    MGLevelObject<LevelMatrixType> transfers_dev(0, max_level - 1);
    for (unsigned int l = 0; l < max_level; ++l)
      transfers_dev[l].reinit(transfers[l]);

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

    // Coarse solve stays on the host: the wrapper copies the coarse residual
    // over and the correction back.
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

    // Right-hand side and solution in the layout of the fine device matrix,
    // so that no copy is needed inside the CG iteration.
    HostVector rhs_layout(level_dev[max_level].get_domain_partitioner());
    rhs_layout.copy_locally_owned_data_from(rhs);
    DeviceVector rhs_dev;
    level_dev[max_level].initialize_dof_vector(rhs_dev);
    rhs_dev.import_elements(rhs_layout, VectorOperation::insert);
    level_dev[max_level].initialize_dof_vector(solution_dev);

    SolverCG<DeviceVector> cg(control);
    cg.solve(level_dev[max_level], solution_dev, rhs_dev, preconditioner);
    pcout << "Device CG+MG: " << control.last_step() << " iterations, residual "
          << std::scientific << std::setprecision(2) << control.last_value()
          << std::defaultfloat << std::endl;
    pcout << "Same iteration count: "
          << (control.last_step() == host_iterations ? "yes" : "no")
          << std::endl;

    // -------------------------------------------------------------------
    // Same V-cycle, but the finest level is the matrix-free device operator
    // instead of the assembled matrix. The levels are wrapped in
    // LinearOperatorMG so that the fine and coarse operators can be of
    // different types.
    // -------------------------------------------------------------------
    DeviceDGOperator<dim, degree> fine_mf;
    fine_mf.reinit(mapping, dof_handler);

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
    DeviceVector rhs_mf;
    fine_mf.initialize_dof_vector(rhs_mf);
    rhs_mf.import_elements(rhs_layout_mf, VectorOperation::insert);
    fine_mf.initialize_dof_vector(solution_mf);

    SolverCG<DeviceVector> cg_mf(control);
    cg_mf.solve(fine_mf, solution_mf, rhs_mf, preconditioner_mf);
    pcout << "Device CG+MG, matrix-free fine level: " << control.last_step()
          << " iterations, residual " << std::scientific << std::setprecision(2)
          << control.last_value() << std::defaultfloat << std::endl;
    pcout << "Same iteration count: "
          << (control.last_step() == host_iterations ? "yes" : "no")
          << std::endl;

    // -------------------------------------------------------------------
    // Same again with the batched sum-factorised operator on the finest
    // level (step-59 penalty, so that it matches the projected levels).
    // -------------------------------------------------------------------
    PortableDG::LaplaceOperatorDG<dim, degree> fine_batched;
    fine_batched.reinit(mapping, dof_handler);

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
    DeviceVector rhs_b;
    fine_batched.initialize_dof_vector(rhs_b);
    rhs_b.import_elements(rhs_layout_b, VectorOperation::insert);
    fine_batched.initialize_dof_vector(solution_batched);

    SolverCG<DeviceVector> cg_b(control);
    cg_b.solve(fine_batched, solution_batched, rhs_b, preconditioner_batched);
    pcout << "Device CG+MG, batched matrix-free fine level: "
          << control.last_step() << " iterations, residual " << std::scientific
          << std::setprecision(2) << control.last_value() << std::defaultfloat
          << std::endl;
    pcout << "Same iteration count: "
          << (control.last_step() == host_iterations ? "yes" : "no")
          << std::endl;
  }

  // Compare the two solutions
  {
    HostVector dev_on_host(solution_dev.get_partitioner());
    dev_on_host.import_elements(solution_dev, VectorOperation::insert);
    HostVector diff(solution_host);
    diff.copy_locally_owned_data_from(dev_on_host);
    diff -= solution_host;
    const double rel = diff.l2_norm() / solution_host.l2_norm();
    pcout << "Solutions agree: " << (rel < 1e-6 ? "OK" : "FAIL") << std::endl;

    HostVector mf_on_host(solution_mf.get_partitioner());
    mf_on_host.import_elements(solution_mf, VectorOperation::insert);
    diff.copy_locally_owned_data_from(mf_on_host);
    diff -= solution_host;
    const double rel_mf = diff.l2_norm() / solution_host.l2_norm();
    pcout << "Solutions agree (matrix-free fine level): "
          << (rel_mf < 1e-6 ? "OK" : "FAIL") << std::endl;

    HostVector b_on_host(solution_batched.get_partitioner());
    b_on_host.import_elements(solution_batched, VectorOperation::insert);
    diff.copy_locally_owned_data_from(b_on_host);
    diff -= solution_host;
    const double rel_b = diff.l2_norm() / solution_host.l2_norm();
    pcout << "Solutions agree (batched matrix-free fine level): "
          << (rel_b < 1e-6 ? "OK" : "FAIL") << std::endl;
  }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm                   comm = MPI_COMM_WORLD;

  run<2, 1>(comm);
  run<2, 2>(comm);
}
