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

#ifndef polydeal_device_sparse_matrix_h
#define polydeal_device_sparse_matrix_h

#include <deal.II/base/config.h>

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/partitioner.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <EpetraExt_Transpose_RowMatrix.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Kokkos_Core.hpp>

#include <memory>
#include <vector>

namespace dealii
{
  /**
   * A distributed sparse matrix in compressed-row (CSR) format whose storage
   * lives in the Kokkos memory space selected by MemorySpace::Default. On a
   * deal.II build with a CUDA or HIP Kokkos backend this is device memory; with
   * a Serial or OpenMP backend it is host memory and the kernels run on the
   * CPU. The class is meant to be a drop-in replacement for the Trilinos
   * matrices used in the agglomerated multigrid pipeline of polyDEAL, so that
   * the whole V-cycle can operate on
   * LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>.
   *
   * The matrix is built once, on the host, from either a
   * TrilinosWrappers::SparseMatrix (distributed) or a dealii::SparseMatrix
   * (serial), and is read-only afterwards.
   *
   * Column indices are stored in the local numbering of a
   * Utilities::MPI::Partitioner whose locally owned range is the domain of the
   * matrix and whose ghost set is the set of off-process columns appearing in
   * the local rows. A matrix-vector product is therefore a ghost update
   * followed by a single kernel over the locally owned rows, with no
   * compress() afterwards.
   *
   * Transposed products use a second CSR, built on the host at setup from the
   * explicitly transposed matrix, so that they follow the same owned-rows-only
   * pattern and need neither atomics nor a compress().
   */
  template <typename Number = double>
  class DeviceCSRMatrix : public EnableObserverPointer
  {
  public:
    using value_type  = Number;
    using size_type   = types::global_dof_index;
    using KokkosSpace = typename MemorySpace::Default::kokkos_space;
    using ExecSpace   = typename KokkosSpace::execution_space;
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
    using PartitionerPtr = std::shared_ptr<const Utilities::MPI::Partitioner>;

    /**
     * Default constructor. Produces an empty matrix; call reinit() before use.
     */
    DeviceCSRMatrix() = default;

    /**
     * Build from a (compressed) Trilinos matrix.
     */
    explicit DeviceCSRMatrix(const TrilinosWrappers::SparseMatrix &matrix);

    /**
     * Build from a serial deal.II sparse matrix. The resulting matrix uses
     * MPI_COMM_SELF.
     */
    explicit DeviceCSRMatrix(const SparseMatrix<Number> &matrix);

    void
    reinit(const TrilinosWrappers::SparseMatrix &matrix);

    void
    reinit(const SparseMatrix<Number> &matrix);

    /**
     * Release all storage.
     */
    void
    clear();

    /**
     * Global number of rows.
     */
    size_type
    m() const;

    /**
     * Global number of columns.
     */
    size_type
    n() const;

    /**
     * Global number of stored entries.
     */
    size_type
    n_nonzero_elements() const;

    /**
     * Number of locally owned rows.
     */
    unsigned int
    local_n_rows() const;

    MPI_Comm
    get_mpi_communicator() const;

    /**
     * Partitioner of the domain: locally owned columns plus the off-process
     * columns referenced by local rows. Vectors with this layout can be used as
     * source of vmult() without any copy.
     */
    const PartitionerPtr &
    get_domain_partitioner() const;

    /**
     * Partitioner of the range: locally owned rows plus the off-process rows
     * referenced by the local rows of the transpose. Vectors with this layout
     * can be used as source of Tvmult() without any copy.
     */
    const PartitionerPtr &
    get_range_partitioner() const;

    /**
     * Initialize @p v with the domain layout. For a square matrix this is the
     * layout expected by both vmult() (source) and Tvmult() (destination).
     */
    void
    initialize_dof_vector(VectorType &v) const;

    void
    initialize_domain_vector(VectorType &v) const;

    void
    initialize_range_vector(VectorType &v) const;

    /**
     * dst = A src. @p dst must have as many locally owned entries as the
     * locally owned rows. @p src must have the same locally owned range as the
     * domain; if its ghost layout differs from get_domain_partitioner() a
     * temporary copy is made.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const;

    void
    vmult_add(VectorType &dst, const VectorType &src) const;

    void
    Tvmult(VectorType &dst, const VectorType &src) const;

    void
    Tvmult_add(VectorType &dst, const VectorType &src) const;

    /**
     * Fill @p diagonal with the diagonal entries of a square matrix. Missing
     * diagonal entries are reported as zero.
     */
    void
    get_diagonal(VectorType &diagonal) const;

    /**
     * Fill @p inverse_diagonal with the reciprocal of the diagonal entries of a
     * square matrix. Zero entries are replaced by one.
     */
    void
    get_inverse_diagonal(VectorType &inverse_diagonal) const;

    /**
     * Element access is not available on the device; this function only
     * exists to satisfy generic interfaces and always throws.
     */
    Number
    el(const size_type row, const size_type col) const;

    std::size_t
    memory_consumption() const;

  private:
    /**
     * The CSR arrays of one (forward or transposed) matrix.
     */
    struct CSR
    {
      Kokkos::View<unsigned int *, KokkosSpace> row_ptr;
      Kokkos::View<unsigned int *, KokkosSpace> col_idx;
      Kokkos::View<Number *, KokkosSpace>       values;
      unsigned int                              n_rows    = 0;
      unsigned int                              n_cols    = 0;
      std::size_t                               n_entries = 0;

      void
      clear();
    };

    /**
     * Row-wise host representation used while building.
     */
    struct HostCSR
    {
      std::vector<unsigned int> row_ptr;
      std::vector<unsigned int> col_idx;
      std::vector<Number>       values;
    };

    /**
     * Read the locally owned rows of @p mat into @p csr. Rows are placed in
     * the local numbering of @p row_partitioner (locally owned part only),
     * columns in the local numbering of a new partitioner whose owned range is
     * @p owned_cols and whose ghosts are the off-process columns of @p mat;
     * that partitioner is returned through @p col_partitioner.
     */
    static void
    fill_from_epetra(const Epetra_CrsMatrix &mat,
                     const IndexSet         &owned_rows,
                     const IndexSet         &owned_cols,
                     const MPI_Comm          comm,
                     CSR                    &csr,
                     PartitionerPtr         &col_partitioner);

    static void
    upload(const HostCSR &host, CSR &csr);

    /**
     * dst (+)= csr * src, with @p src in the layout of @p col_partitioner.
     */
    void
    apply(const CSR            &csr,
          const PartitionerPtr &col_partitioner,
          VectorType           &tmp,
          VectorType           &dst,
          const VectorType     &src,
          const bool            add) const;

    CSR forward;
    CSR transposed;

    /**
     * Columns of the forward matrix: owned domain + ghost columns.
     */
    PartitionerPtr domain_partitioner;

    /**
     * Columns of the transposed matrix: owned range + ghost rows.
     */
    PartitionerPtr range_partitioner;

    MPI_Comm  communicator     = MPI_COMM_SELF;
    size_type n_global_rows    = 0;
    size_type n_global_cols    = 0;
    size_type n_global_entries = 0;

    /**
     * Scratch vectors used when the source of a product does not have the
     * expected ghost layout.
     */
    mutable VectorType tmp_domain;
    mutable VectorType tmp_range;
  };



  // ------------------------------ implementation ---------------------------

  template <typename Number>
  void
  DeviceCSRMatrix<Number>::CSR::clear()
  {
    row_ptr   = Kokkos::View<unsigned int *, KokkosSpace>();
    col_idx   = Kokkos::View<unsigned int *, KokkosSpace>();
    values    = Kokkos::View<Number *, KokkosSpace>();
    n_rows    = 0;
    n_cols    = 0;
    n_entries = 0;
  }



  template <typename Number>
  DeviceCSRMatrix<Number>::DeviceCSRMatrix(
    const TrilinosWrappers::SparseMatrix &matrix)
  {
    reinit(matrix);
  }



  template <typename Number>
  DeviceCSRMatrix<Number>::DeviceCSRMatrix(const SparseMatrix<Number> &matrix)
  {
    reinit(matrix);
  }



  template <typename Number>
  void
  DeviceCSRMatrix<Number>::clear()
  {
    forward.clear();
    transposed.clear();
    domain_partitioner.reset();
    range_partitioner.reset();
    tmp_domain.reinit(0);
    tmp_range.reinit(0);
    communicator     = MPI_COMM_SELF;
    n_global_rows    = 0;
    n_global_cols    = 0;
    n_global_entries = 0;
  }



  template <typename Number>
  void
  DeviceCSRMatrix<Number>::upload(const HostCSR &host, CSR &csr)
  {
    AssertThrow(host.col_idx.size() <
                  static_cast<std::size_t>(
                    std::numeric_limits<unsigned int>::max()),
                ExcMessage("Too many local entries for 32-bit offsets."));

    csr.n_entries = host.col_idx.size();
    csr.row_ptr   = Kokkos::View<unsigned int *, KokkosSpace>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "polydeal::csr_row_ptr"),
      host.row_ptr.size());
    csr.col_idx = Kokkos::View<unsigned int *, KokkosSpace>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "polydeal::csr_col_idx"),
      host.col_idx.size());
    csr.values = Kokkos::View<Number *, KokkosSpace>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "polydeal::csr_values"),
      host.values.size());

    using HostUnmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    Kokkos::deep_copy(
      csr.row_ptr,
      Kokkos::View<const unsigned int *, Kokkos::HostSpace, HostUnmanaged>(
        host.row_ptr.data(), host.row_ptr.size()));
    Kokkos::deep_copy(
      csr.col_idx,
      Kokkos::View<const unsigned int *, Kokkos::HostSpace, HostUnmanaged>(
        host.col_idx.data(), host.col_idx.size()));
    Kokkos::deep_copy(
      csr.values,
      Kokkos::View<const Number *, Kokkos::HostSpace, HostUnmanaged>(
        host.values.data(), host.values.size()));
    Kokkos::fence();
  }



  template <typename Number>
  void
  DeviceCSRMatrix<Number>::fill_from_epetra(const Epetra_CrsMatrix &mat,
                                            const IndexSet         &owned_rows,
                                            const IndexSet         &owned_cols,
                                            const MPI_Comm          comm,
                                            CSR                    &csr,
                                            PartitionerPtr &col_partitioner)
  {
    AssertThrow(mat.Filled(),
                ExcMessage("The Trilinos matrix must be compressed."));

    const int n_local_rows = mat.NumMyRows();
    const int n_local_cols = mat.NumMyCols();
    AssertDimension(static_cast<size_type>(n_local_rows),
                    owned_rows.n_elements());

    // Column partitioner: owned columns first, then the off-process columns
    // referenced by the local rows, in increasing global order.
    const Epetra_Map &col_map = mat.ColMap();
    const Epetra_Map &row_map = mat.RowMap();

    IndexSet ghost_cols(owned_cols.size());
    for (int lid = 0; lid < n_local_cols; ++lid)
      {
        const size_type gid = static_cast<size_type>(col_map.GID64(lid));
        if (!owned_cols.is_element(gid))
          ghost_cols.add_index(gid);
      }
    ghost_cols.compress();
    col_partitioner = std::make_shared<Utilities::MPI::Partitioner>(owned_cols,
                                                                    ghost_cols,
                                                                    comm);

    std::vector<unsigned int> epetra_col_to_local(n_local_cols);
    for (int lid = 0; lid < n_local_cols; ++lid)
      epetra_col_to_local[lid] = col_partitioner->global_to_local(
        static_cast<size_type>(col_map.GID64(lid)));

    // Rows: Epetra's local row order may differ from the partitioner's, so
    // place each row explicitly.
    Utilities::MPI::Partitioner row_partitioner(owned_rows,
                                                IndexSet(owned_rows.size()),
                                                comm);
    std::vector<unsigned int>   epetra_row_to_local(n_local_rows);
    for (int lid = 0; lid < n_local_rows; ++lid)
      epetra_row_to_local[lid] = row_partitioner.global_to_local(
        static_cast<size_type>(row_map.GID64(lid)));

    HostCSR host;
    host.row_ptr.assign(n_local_rows + 1, 0);
    for (int lid = 0; lid < n_local_rows; ++lid)
      host.row_ptr[epetra_row_to_local[lid] + 1] = mat.NumMyEntries(lid);
    for (int r = 0; r < n_local_rows; ++r)
      host.row_ptr[r + 1] += host.row_ptr[r];

    host.col_idx.resize(host.row_ptr.back());
    host.values.resize(host.row_ptr.back());
    for (int lid = 0; lid < n_local_rows; ++lid)
      {
        int       n_entries;
        double   *values;
        int      *indices;
        const int ierr = mat.ExtractMyRowView(lid, n_entries, values, indices);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));

        unsigned int pos = host.row_ptr[epetra_row_to_local[lid]];
        for (int k = 0; k < n_entries; ++k, ++pos)
          {
            host.col_idx[pos] = epetra_col_to_local[indices[k]];
            host.values[pos]  = static_cast<Number>(values[k]);
          }
      }

    csr.n_rows = n_local_rows;
    csr.n_cols = col_partitioner->locally_owned_size() +
                 col_partitioner->n_ghost_indices();
    upload(host, csr);
  }



  template <typename Number>
  void
  DeviceCSRMatrix<Number>::reinit(const TrilinosWrappers::SparseMatrix &matrix)
  {
    clear();

    communicator  = matrix.get_mpi_communicator();
    n_global_rows = matrix.m();
    n_global_cols = matrix.n();

    const IndexSet owned_rows = matrix.locally_owned_range_indices();
    const IndexSet owned_cols = matrix.locally_owned_domain_indices();

    // Forward matrix: rows in the range, columns in the domain.
    fill_from_epetra(matrix.trilinos_matrix(),
                     owned_rows,
                     owned_cols,
                     communicator,
                     forward,
                     domain_partitioner);

    // Transposed matrix: rows in the domain, columns in the range. The
    // transpose is formed on the host by EpetraExt, which redistributes the
    // entries so that every rank owns the rows of A^T that correspond to its
    // owned domain indices.
    {
      EpetraExt::RowMatrix_Transpose transposer;
      Epetra_RowMatrix              &transposed_row_matrix =
        transposer(const_cast<Epetra_CrsMatrix &>(matrix.trilinos_matrix()));
      const Epetra_CrsMatrix &transposed_matrix =
        dynamic_cast<const Epetra_CrsMatrix &>(transposed_row_matrix);

      fill_from_epetra(transposed_matrix,
                       owned_cols,
                       owned_rows,
                       communicator,
                       transposed,
                       range_partitioner);
    }

    n_global_entries =
      Utilities::MPI::sum(static_cast<size_type>(forward.n_entries),
                          communicator);
  }



  template <typename Number>
  void
  DeviceCSRMatrix<Number>::reinit(const SparseMatrix<Number> &matrix)
  {
    clear();

    communicator  = MPI_COMM_SELF;
    n_global_rows = matrix.m();
    n_global_cols = matrix.n();

    const IndexSet owned_rows = complete_index_set(matrix.m());
    const IndexSet owned_cols = complete_index_set(matrix.n());
    domain_partitioner =
      std::make_shared<Utilities::MPI::Partitioner>(owned_cols,
                                                    IndexSet(matrix.n()),
                                                    communicator);
    range_partitioner =
      std::make_shared<Utilities::MPI::Partitioner>(owned_rows,
                                                    IndexSet(matrix.m()),
                                                    communicator);

    const unsigned int n_rows = matrix.m();
    const unsigned int n_cols = matrix.n();

    HostCSR host;
    host.row_ptr.assign(n_rows + 1, 0);
    for (unsigned int r = 0; r < n_rows; ++r)
      host.row_ptr[r + 1] =
        host.row_ptr[r] + matrix.get_sparsity_pattern().row_length(r);
    host.col_idx.reserve(host.row_ptr.back());
    host.values.reserve(host.row_ptr.back());
    for (unsigned int r = 0; r < n_rows; ++r)
      for (auto it = matrix.begin(r); it != matrix.end(r); ++it)
        {
          host.col_idx.push_back(it->column());
          host.values.push_back(it->value());
        }

    forward.n_rows = n_rows;
    forward.n_cols = n_cols;
    upload(host, forward);

    // Transpose on the host.
    HostCSR host_t;
    host_t.row_ptr.assign(n_cols + 1, 0);
    for (const unsigned int c : host.col_idx)
      ++host_t.row_ptr[c + 1];
    for (unsigned int c = 0; c < n_cols; ++c)
      host_t.row_ptr[c + 1] += host_t.row_ptr[c];
    host_t.col_idx.resize(host.col_idx.size());
    host_t.values.resize(host.values.size());
    std::vector<unsigned int> fill_pos(host_t.row_ptr.begin(),
                                       host_t.row_ptr.end() - 1);
    for (unsigned int r = 0; r < n_rows; ++r)
      for (unsigned int k = host.row_ptr[r]; k < host.row_ptr[r + 1]; ++k)
        {
          const unsigned int c   = host.col_idx[k];
          const unsigned int pos = fill_pos[c]++;
          host_t.col_idx[pos]    = r;
          host_t.values[pos]     = host.values[k];
        }

    transposed.n_rows = n_cols;
    transposed.n_cols = n_rows;
    upload(host_t, transposed);

    n_global_entries = forward.n_entries;
  }



  template <typename Number>
  inline typename DeviceCSRMatrix<Number>::size_type
  DeviceCSRMatrix<Number>::m() const
  {
    return n_global_rows;
  }



  template <typename Number>
  inline typename DeviceCSRMatrix<Number>::size_type
  DeviceCSRMatrix<Number>::n() const
  {
    return n_global_cols;
  }



  template <typename Number>
  inline typename DeviceCSRMatrix<Number>::size_type
  DeviceCSRMatrix<Number>::n_nonzero_elements() const
  {
    return n_global_entries;
  }



  template <typename Number>
  inline unsigned int
  DeviceCSRMatrix<Number>::local_n_rows() const
  {
    return forward.n_rows;
  }



  template <typename Number>
  inline MPI_Comm
  DeviceCSRMatrix<Number>::get_mpi_communicator() const
  {
    return communicator;
  }



  template <typename Number>
  inline const typename DeviceCSRMatrix<Number>::PartitionerPtr &
  DeviceCSRMatrix<Number>::get_domain_partitioner() const
  {
    return domain_partitioner;
  }



  template <typename Number>
  inline const typename DeviceCSRMatrix<Number>::PartitionerPtr &
  DeviceCSRMatrix<Number>::get_range_partitioner() const
  {
    return range_partitioner;
  }



  template <typename Number>
  void
  DeviceCSRMatrix<Number>::initialize_domain_vector(VectorType &v) const
  {
    Assert(domain_partitioner, ExcNotInitialized());
    v.reinit(domain_partitioner);
  }



  template <typename Number>
  void
  DeviceCSRMatrix<Number>::initialize_range_vector(VectorType &v) const
  {
    Assert(range_partitioner, ExcNotInitialized());
    v.reinit(range_partitioner);
  }



  template <typename Number>
  void
  DeviceCSRMatrix<Number>::initialize_dof_vector(VectorType &v) const
  {
    initialize_domain_vector(v);
  }



  template <typename Number>
  void
  DeviceCSRMatrix<Number>::apply(const CSR            &csr,
                                 const PartitionerPtr &col_partitioner,
                                 VectorType           &tmp,
                                 VectorType           &dst,
                                 const VectorType     &src,
                                 const bool            add) const
  {
    Assert(col_partitioner, ExcNotInitialized());
    AssertDimension(dst.locally_owned_size(), csr.n_rows);
    AssertDimension(src.locally_owned_size(),
                    col_partitioner->locally_owned_size());

    // Bring the source into the expected ghost layout.
    const VectorType *src_ptr = &src;
    if (src.get_partitioner().get() != col_partitioner.get() &&
        !src.get_partitioner()->is_compatible(*col_partitioner))
      {
        if (tmp.get_partitioner().get() != col_partitioner.get())
          tmp.reinit(col_partitioner);
        tmp.copy_locally_owned_data_from(src);
        src_ptr = &tmp;
      }

    const bool had_ghosts = src_ptr->has_ghost_elements();
    if (!had_ghosts)
      src_ptr->update_ghost_values();

    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    const Kokkos::View<const Number *, KokkosSpace, Unmanaged> src_view(
      src_ptr->get_values(), csr.n_cols);
    Kokkos::View<Number *, KokkosSpace, Unmanaged> dst_view(dst.get_values(),
                                                            csr.n_rows);

    // Local copies for capture by value in the kernel.
    const auto row_ptr = csr.row_ptr;
    const auto col_idx = csr.col_idx;
    const auto values  = csr.values;

    Kokkos::parallel_for(
      "polydeal::DeviceCSRMatrix::apply",
      Kokkos::RangePolicy<ExecSpace>(0, csr.n_rows),
      KOKKOS_LAMBDA(const unsigned int row) {
        Number sum = 0;
        for (unsigned int k = row_ptr(row); k < row_ptr(row + 1); ++k)
          sum += values(k) * src_view(col_idx(k));
        if (add)
          dst_view(row) += sum;
        else
          dst_view(row) = sum;
      });
    Kokkos::fence();

    if (!had_ghosts)
      src_ptr->zero_out_ghost_values();
  }



  template <typename Number>
  void
  DeviceCSRMatrix<Number>::vmult(VectorType &dst, const VectorType &src) const
  {
    apply(forward, domain_partitioner, tmp_domain, dst, src, false);
  }



  template <typename Number>
  void
  DeviceCSRMatrix<Number>::vmult_add(VectorType       &dst,
                                     const VectorType &src) const
  {
    apply(forward, domain_partitioner, tmp_domain, dst, src, true);
  }



  template <typename Number>
  void
  DeviceCSRMatrix<Number>::Tvmult(VectorType &dst, const VectorType &src) const
  {
    apply(transposed, range_partitioner, tmp_range, dst, src, false);
  }



  template <typename Number>
  void
  DeviceCSRMatrix<Number>::Tvmult_add(VectorType       &dst,
                                      const VectorType &src) const
  {
    apply(transposed, range_partitioner, tmp_range, dst, src, true);
  }



  template <typename Number>
  void
  DeviceCSRMatrix<Number>::get_diagonal(VectorType &diagonal) const
  {
    Assert(domain_partitioner && range_partitioner, ExcNotInitialized());
    Assert(n_global_rows == n_global_cols &&
             domain_partitioner->locally_owned_range() ==
               range_partitioner->locally_owned_range(),
           ExcMessage("The diagonal is only defined for square matrices "
                      "whose rows and columns are partitioned identically."));
    AssertDimension(diagonal.locally_owned_size(), forward.n_rows);

    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    Kokkos::View<Number *, KokkosSpace, Unmanaged> diag_view(
      diagonal.get_values(), forward.n_rows);

    const auto row_ptr = forward.row_ptr;
    const auto col_idx = forward.col_idx;
    const auto values  = forward.values;

    Kokkos::parallel_for(
      "polydeal::DeviceCSRMatrix::get_diagonal",
      Kokkos::RangePolicy<ExecSpace>(0, forward.n_rows),
      KOKKOS_LAMBDA(const unsigned int row) {
        Number d = 0;
        for (unsigned int k = row_ptr(row); k < row_ptr(row + 1); ++k)
          if (col_idx(k) == row)
            d += values(k);
        diag_view(row) = d;
      });
    Kokkos::fence();
  }



  template <typename Number>
  void
  DeviceCSRMatrix<Number>::get_inverse_diagonal(
    VectorType &inverse_diagonal) const
  {
    get_diagonal(inverse_diagonal);

    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    Kokkos::View<Number *, KokkosSpace, Unmanaged> diag_view(
      inverse_diagonal.get_values(), forward.n_rows);

    Kokkos::parallel_for(
      "polydeal::DeviceCSRMatrix::get_inverse_diagonal",
      Kokkos::RangePolicy<ExecSpace>(0, forward.n_rows),
      KOKKOS_LAMBDA(const unsigned int row) {
        const Number d = diag_view(row);
        diag_view(row) = (d == Number(0)) ? Number(1) : Number(1) / d;
      });
    Kokkos::fence();
  }



  template <typename Number>
  Number
  DeviceCSRMatrix<Number>::el(const size_type, const size_type) const
  {
    AssertThrow(false,
                ExcMessage("DeviceCSRMatrix does not provide element access."));
    return Number();
  }



  template <typename Number>
  std::size_t
  DeviceCSRMatrix<Number>::memory_consumption() const
  {
    const auto csr_bytes = [](const CSR &csr) {
      return csr.row_ptr.extent(0) * sizeof(unsigned int) +
             csr.col_idx.extent(0) * sizeof(unsigned int) +
             csr.values.extent(0) * sizeof(Number);
    };
    return csr_bytes(forward) + csr_bytes(transposed);
  }

} // namespace dealii

#endif
