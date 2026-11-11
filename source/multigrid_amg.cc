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

#include <deal.II/lac/trilinos_vector.h>

#include <device_sparse_matrix.h>
#include <multigrid_amg.h>

namespace dealii
{
  namespace internal
  {
    namespace MGTransferAgglomerationImplementation
    {
      template <typename T>
      struct is_distributed_vector : std::false_type
      {};

      template <typename Number, typename MemorySpace>
      struct is_distributed_vector<
        LinearAlgebra::distributed::Vector<Number, MemorySpace>>
        : std::true_type
      {};
    } // namespace MGTransferAgglomerationImplementation
  } // namespace internal



  template <int dim, typename VectorType, typename MatrixType>
  MGTransferAgglomeration<dim, VectorType, MatrixType>::MGTransferAgglomeration(
    const MGLevelObject<MatrixType *>    &transfer_matrices_,
    const std::vector<DoFHandler<dim> *> &dof_handlers_)
  {
    Assert(transfer_matrices_.n_levels() > 0, ExcInternalError());
    Assert(transfer_matrices_.max_level() < dof_handlers_.size(),
           ExcMessage("There must be one more DoFHandler than transfers."));
    transfer_matrices.resize(0, dof_handlers_.size());
    dof_handlers.assign(dof_handlers_.begin(), dof_handlers_.end());

    for (unsigned int l = transfer_matrices_.min_level();
         l <= transfer_matrices_.max_level();
         ++l)
      transfer_matrices[l] = transfer_matrices_[l];
  }



  template <int dim, typename VectorType, typename MatrixType>
  MGTransferAgglomeration<dim, VectorType, MatrixType>::MGTransferAgglomeration(
    const MGLevelObject<MatrixType>      &transfer_matrices_,
    const std::vector<DoFHandler<dim> *> &dof_handlers_)
  {
    Assert(transfer_matrices_.n_levels() > 0, ExcInternalError());
    Assert(transfer_matrices_.max_level() < dof_handlers_.size(),
           ExcMessage("There must be one more DoFHandler than transfers."));
    transfer_matrices.resize(0, dof_handlers_.size());
    dof_handlers.assign(dof_handlers_.begin(), dof_handlers_.end());

    for (unsigned int l = transfer_matrices_.min_level();
         l <= transfer_matrices_.max_level();
         ++l)
      transfer_matrices[l] = const_cast<MatrixType *>(&transfer_matrices_[l]);
  }



  template <int dim, typename VectorType, typename MatrixType>
  void
  MGTransferAgglomeration<dim, VectorType, MatrixType>::prolongate(
    const unsigned int to_level,
    VectorType        &dst,
    const VectorType  &src) const
  {
    dst = typename VectorType::value_type(0.0);
    prolongate_and_add(to_level, dst, src);
  }



  template <int dim, typename VectorType, typename MatrixType>
  void
  MGTransferAgglomeration<dim, VectorType, MatrixType>::prolongate_and_add(
    const unsigned int to_level,
    VectorType        &dst,
    const VectorType  &src) const
  {
    Assert(transfer_matrices[to_level - 1] != nullptr,
           ExcMessage("Transfer matrix has not been initialized."));
    transfer_matrices[to_level - 1]->vmult_add(dst, src);
  }



  template <int dim, typename VectorType, typename MatrixType>
  void
  MGTransferAgglomeration<dim, VectorType, MatrixType>::restrict_and_add(
    const unsigned int from_level,
    VectorType        &dst,
    const VectorType  &src) const
  {
    Assert(transfer_matrices[from_level - 1] != nullptr,
           ExcMessage("Matrix has not been initialized."));
    transfer_matrices[from_level - 1]->Tvmult_add(dst, src);
  }



  template <int dim, typename VectorType, typename MatrixType>
  void
  MGTransferAgglomeration<dim, VectorType, MatrixType>::copy_to_mg(
    const DoFHandler<dim>     &dof_handler,
    MGLevelObject<VectorType> &dst,
    const VectorType          &src) const
  {
    (void)dof_handler; // required by interface, but not needed.
    for (unsigned int level = dst.min_level(); level <= dst.max_level();
         ++level)
      dst[level].reinit(dof_handlers[level]->locally_owned_dofs(),
                        dof_handlers[level]->get_mpi_communicator());

    if constexpr (std::is_same_v<VectorType, TrilinosWrappers::MPI::Vector>)
      dst[dst.max_level()] = src;
    else if constexpr (internal::MGTransferAgglomerationImplementation::
                         is_distributed_vector<VectorType>::value)
      dst[dst.max_level()].copy_locally_owned_data_from(src);
    else
      DEAL_II_NOT_IMPLEMENTED();
  }



  template <int dim, typename VectorType, typename MatrixType>
  void
  MGTransferAgglomeration<dim, VectorType, MatrixType>::copy_from_mg(
    const DoFHandler<dim>           &dof_handler,
    VectorType                      &dst,
    const MGLevelObject<VectorType> &src) const
  {
    (void)dof_handler;
    if constexpr (std::is_same_v<VectorType, TrilinosWrappers::MPI::Vector>)
      dst = src[src.max_level()];
    else if constexpr (internal::MGTransferAgglomerationImplementation::
                         is_distributed_vector<VectorType>::value)
      dst.copy_locally_owned_data_from(src[src.max_level()]);
    else
      DEAL_II_NOT_IMPLEMENTED();
  }



  // Explicit instantiations: host vectors with Trilinos transfer matrices.
  template class MGTransferAgglomeration<
    1,
    LinearAlgebra::distributed::Vector<double>>;
  template class MGTransferAgglomeration<
    2,
    LinearAlgebra::distributed::Vector<double>>;
  template class MGTransferAgglomeration<
    3,
    LinearAlgebra::distributed::Vector<double>>;

  template class MGTransferAgglomeration<1, TrilinosWrappers::MPI::Vector>;
  template class MGTransferAgglomeration<2, TrilinosWrappers::MPI::Vector>;
  template class MGTransferAgglomeration<3, TrilinosWrappers::MPI::Vector>;

  // Device vectors with device CSR transfer matrices.
  template class MGTransferAgglomeration<
    1,
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>,
    DeviceCSRMatrix<double>>;
  template class MGTransferAgglomeration<
    2,
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>,
    DeviceCSRMatrix<double>>;
  template class MGTransferAgglomeration<
    3,
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>,
    DeviceCSRMatrix<double>>;
} // namespace dealii
