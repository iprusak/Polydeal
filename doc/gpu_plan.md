# GPU integration plan for polyDEAL

Status: November 2026. Phases 0-3 are implemented on branch `gpu/phase1-device-csr`: `DeviceCSRMatrix` (`include/device_sparse_matrix.h`, test `device_csr_vmult`), the matrix-type-generic `MGTransferAgglomeration`, a device-aware `Utils::MGCoarseDirect`, the batched sum-factorised `PortableDG::LaplaceOperatorDG` (`include/portable/`, kernels taken from dealiiX-portable-multigrid) as the matrix-free fine level, the point-wise `DeviceDGOperator` (`include/device_dg_operator.h`, kept for the polytopal case), the tests `device_dg_operator` and `device_agglomerated_mg`, and the example `examples/agglo_mg_device.cc`. Phase 4 is a proposal.

## 1. Verified starting point

| Fact | Where checked |
|---|---|
| No GPU, no `nvcc`/`hipcc` on this machine | `nvidia-smi`, `which nvcc` |
| Every local Kokkos build is Serial only (the Trilinos one also has OpenMP) | `KokkosCore_config.h` of each install |
| `build/` points at `dealii-mg-transfer`, which has Trilinos OFF, so the existing build never configured | `build/CMakeCache.txt` |
| polyDEAL configures cleanly against `~/packages/dealii/dealii-trilinos` (9.8.0, Kokkos 4.2.1, Trilinos ON, Tpetra OFF), `dealii-9.7.1`, `dealii-9.8.0-pre`, `dealii-gabriel` | scratch configure of each |
| `Portable::MatrixFree` has **no face loops** in 9.7, 9.8 or 9.9-master. Only cell loops. | grep of `portable_matrix_free.h` |
| deal.II device building blocks that do exist: `LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>` with ghost exchange, `SolverCG` on such vectors, `PreconditionChebyshev` + `DiagonalMatrix` specialised for `MemorySpace::Default`, `Portable::FEEvaluation` cell kernels (step-64), `ReadWriteVector::import_elements` host/device copies, `Portable::MGTransferGlobalCoarsening` (FE_Q/FE_DGQ nested transfers only). | headers of `dealii-master` |
| `TpetraWrappers::SparseMatrix<Number, MemorySpace>` exists in 9.9-master, but all local Trilinos builds have Tpetra OFF | `deal.IIConfig.cmake` |
| polyDEAL today uses `LinearAlgebra::distributed::Vector<double>` (host) everywhere in the multigrid path; `MGTransferAgglomeration` is instantiated only for that type; level/transfer matrices are Epetra (`TrilinosWrappers::SparseMatrix`) | `multigrid_amg.{h,cc}`, `utils.h` |

Consequences:

* With Kokkos Serial, `MemorySpace::Default` resolves to host memory. Device code paths therefore compile and run here, which is enough for correctness tests. Performance measurements need a cluster with a CUDA or HIP deal.II + Kokkos build.
* The SIP-DG fine operator (`Utils::MatrixFreeOperators::LaplaceOperatorDG`) cannot be ported 1:1 to `Portable::MatrixFree` because of the missing face loops.
* Epetra objects can never live on a device. Everything that is a Trilinos matrix today must be copied once, at setup, into a device representation.

## 2. Scope decision

Phase 1 and 2 put the **solve** on the device and leave the **setup** on the host:

* host: agglomeration (R-tree/METIS), polytope connectivity and ghost exchange, polytopal assembly via `reinit()`, injection matrices, Galerkin products P^T A P;
* device: every operation inside the multigrid V-cycle and the outer Krylov solver.

Rationale: setup runs once per mesh and touches every pointer-heavy structure in `AgglomerationHandler`; the solve is repeated and only needs sparse matrix-vector products, vector updates and a diagonal. None of the polytopal data structures change.

Phase 3 (fine-level matrix-free DG on device) and Phase 4 (polytopal assembly/operator on device) are listed afterwards with their open problems.

## 3. Target architecture (Phases 1-2)

```
host                                     device (MemorySpace::Default)
----                                     -----------------------------
AgglomerationHandler per level
fill_injection_matrix  -> Epetra P_l  --copy-->  DeviceCSRMatrix P_l, P_l^T
fine SIP matrix A_L    (Epetra)       --copy-->  DeviceCSRMatrix A_L
AmgProjector: A_l = P^T A_{l+1} P     --copy-->  DeviceCSRMatrix A_l, diag(A_l)^-1
coarse direct solver (MUMPS/KLU)      <--copy--  MGCoarseDirectDevice (vector round trip)
                                                 PreconditionChebyshev<DeviceCSRMatrix, Vec>
                                                 MGTransferAgglomeration<dim, Vec>
                                                 Multigrid<Vec> + PreconditionMG
                                                 SolverCG<Vec>
```

`Vec` = `LinearAlgebra::distributed::Vector<double, MemorySpace::Default>`.

### 3.1 `DeviceCSRMatrix<Number>` (new, `include/device_sparse_matrix.h`)

* Storage: `Kokkos::View<unsigned int*>` row offsets, `Kokkos::View<unsigned int*>` local column indices, `Kokkos::View<Number*>` values, all in `MemorySpace::Default::kokkos_space`; a `std::shared_ptr<const Utilities::MPI::Partitioner>` for the domain and one for the range.
* Column indexing uses the "locally owned followed by ghosts" numbering of the partitioner, so a `vmult` is `src.update_ghost_values()` followed by one `Kokkos::parallel_for` over owned rows.
* `Tvmult` is implemented with a second, explicitly transposed CSR built on the host. This avoids atomics and the `compress(add)` on ghost rows.
* Constructors from `TrilinosWrappers::SparseMatrix` (via `Epetra_CrsMatrix::ExtractMyRowView`, column map gives ghost columns directly) and from `dealii::SparseMatrix<Number>` (serial tests).
* `get_diagonal_inverse(Vec&)` for Chebyshev.
* Interface mirrors what `PreconditionChebyshev`, `mg::Matrix`, `LinearOperatorMG` and `SolverCG` need: `vmult`, `vmult_add`, `Tvmult`, `Tvmult_add`, `m()`, `n()`, `initialize_dof_vector`, `get_mpi_communicator`.

Why not Tpetra: all local Trilinos builds have Tpetra OFF and a GPU-enabled Trilinos is a large rebuild; a CSR SpMV is ~150 lines, dependency-free and can be swapped for Tpetra later behind the same interface.

### 3.2 `MGTransferAgglomeration` (modify `multigrid_amg.{h,cc}`)

* Template on `MatrixType` in addition to `VectorType` (today it hard-codes `TrilinosWrappers::SparseMatrix*`).
* Add the explicit instantiation for `Vec` + `DeviceCSRMatrix<double>`.
* `copy_to_mg` / `copy_from_mg`: keep the existing fine-vector copy; it already only uses vector operations.

### 3.3 Level operators

* Coarse levels `0..L-1`: Galerkin products stay on host (`AmgProjector::compute_level_matrices` with Epetra matmul), then each result is copied into a `DeviceCSRMatrix`. Cost is setup-only.
* Fine level `L`, Phase 1: the assembled fine SIP matrix (already produced by `LaplaceOperatorDG::get_system_matrix()`) copied into a `DeviceCSRMatrix`. Memory heavy but correct and simple.
* Fine level `L`, Phase 3: matrix-free device operator, see section 5.

### 3.4 Smoother, coarse solve, outer solver

* `PreconditionChebyshev<DeviceCSRMatrix<double>, Vec>` with `DiagonalMatrix<Vec>`; deal.II has the specialised device path.
* `Utils::MGCoarseDirect` gets a device variant: copy the coarse residual to a host vector with `ReadWriteVector`, solve with the existing Trilinos direct solver, copy back. The coarsest level is small so the round trip is negligible.
* `Multigrid<Vec>`, `PreconditionMG<dim, Vec, MGTransferAgglomeration<...>>`, `SolverCG<Vec>` are vector-generic; verify they compile with `Vec` (expected yes, step-64 uses `SolverCG<Vec>`).
* `LinearOperatorMG` wraps anything with `std::function`, so the level-operator abstraction can stay if useful for the Chebyshev eigenvalue estimate.

## 4. Phased work plan

### Phase 0: build baseline (half a day)

1. Reconfigure `build/` against `~/packages/dealii/dealii-trilinos` (or `dealii-9.7.1`), build the library and examples, run `ctest` to record the current pass/fail state. Nothing has been built from this checkout so far.
2. No CMake option is needed for the device path: Kokkos is a hard dependency of deal.II >= 9.5 and `Utilities::MPI::MPI_InitFinalize` initialises it. Keep the code memory-space-generic so the same binaries run on CPU (Serial/OpenMP) and GPU (CUDA/HIP) builds of deal.II.
3. Optionally build one deal.II with Kokkos OpenMP locally to exercise real `parallel_for` concurrency before going to a cluster.

### Phase 1: device sparse matrix (1 week)

* Implement `DeviceCSRMatrix` as in 3.1.
* Test `test/polydeal/device_csr_vmult.cc` (mpirun=3): build an injection matrix and a fine SIP matrix with existing code, compare `vmult`/`Tvmult` of the Epetra and device versions to 1e-12. Serial variant from `SparseMatrix<double>`.

### Phase 2: device agglomerated multigrid (2 weeks)

* Generalise `MGTransferAgglomeration`, add device `MGCoarseDirect`.
* New example `examples/agglo_mg_device.cc`, a port of `matrix_free_agglo.cc` where the fine level is the assembled matrix on device, Chebyshev smoother on device, CG on device.
* Test `test/polydeal/device_agglomerated_mg.cc` (mpirun=3): same R-tree levels as `coarse_operator_from_matrix_free.cc`; check that CG+MG converges to the same residual with the same iteration count as the host pipeline (numdiff tolerance on the residual).
* Deliverable: a V-cycle whose only host work is the coarse direct solve.

### Phase 3: matrix-free fine level on device (implemented)

`Portable::MatrixFree` has no face loops, so the fine SIP-DG operator comes from the batched Kokkos kernels of the
dealiiX-portable-multigrid project (`include/kernels/laplace_dg_kokkos_kernels.h` there), copied into
`include/portable/dg_kokkos_kernels.h` with only a namespace rename and reformatting. The host side was rewritten
as `PortableDG::LaplaceOperatorDG<dim, degree, Number>` in `include/portable/laplace_operator_dg.h`:

* **Kernels unchanged.** Sum-factorised cell evaluation in collocation form, batches of cells per team sized from a
  shared-memory budget, face traces of `u` and `dn u` computed once per cell and consumed by separate interior- and
  boundary-face kernels, then tested back with the cell basis. Interior faces are visited once; the rows of ghost
  cells are written and discarded (`zero_out_ghost_values()`), so no `compress()` is needed.
* **Host side decoupled.** The operator needs only a `Mapping` and a `DoFHandler`: the partitioner is built from
  the locally owned and relevant DoFs, shape data comes from deal.II's `ShapeInfo`, there is no `MatrixFree`
  object, no constraints and no operator base classes. It offers the same plain interface as `DeviceCSRMatrix`
  (`vmult`, `vmult_add`, `Tvmult`, `initialize_dof_vector`, `get_domain_partitioner`, `get_diagonal`,
  `get_inverse_diagonal`, `m`, `n`, `el`), copies vectors into its layout when theirs differs, and checks at setup
  that both sides of every face see the same quadrature points (non-standard face orientation is not supported).
* **Penalty convention.** The original kernels use sigma = p(p+1)/2 (1/h_- + 1/h_+) with h = |cell|/|face|
  (`PenaltyConvention::cell_extent`). Polydeal's host operator and hence the Galerkin-projected coarse levels use
  the step-59 form, which is exactly twice that on Cartesian cells. `PenaltyConvention::step59` (the default)
  reproduces the host operator to rounding on general meshes, so the fine level is consistent with the hierarchy.
  All boundary faces are Dirichlet by default, as in the host operator.
* **Diagonal.** The exact SIP diagonal (cell and face terms) is assembled on the host at setup and copied over;
  the original `compute_diagonal()` covered cell terms only.
* **Verification** (`device_dg_operator`, 3 ranks, Cartesian and randomly distorted meshes, 2D and 3D, p = 1, 2):
  products, transposed and accumulating products, foreign-layout source and destination, ghosted source, diagonal
  and inverse diagonal against the assembled Trilinos matrix at 1e-11; for the `cell_extent` convention symmetry,
  positivity and the identity A_step59 - A_extent = penalty difference assembled on the host.
* **In the V-cycle** (`device_agglomerated_mg`, `agglo_mg_device.cc`): wrapped in `LinearOperatorMG` next to the
  `DeviceCSRMatrix` coarse levels; iteration counts and residuals identical to the host pipeline.

`DeviceDGOperator` (`include/device_dg_operator.h`), the first attempt written before the batched kernels were
known, stays in the repository for one reason: it evaluates the basis at arbitrary stored reference points, which
the polytopal operator of Phase 4 needs and sum factorisation cannot provide. It is not the fine level of choice.

Measured in `agglo_mg_device.cc` (3D, p=2, 4096 cells, 110592 DoFs, 3 ranks, OpenMP Kokkos backend, one thread
per team):

| fine level | device storage | CG+MG time |
|---|---|---|
| assembled matrix (+ transpose) | 454 MiB | 1.65 s |
| `DeviceDGOperator` (point-wise) | 59 MiB | 6.6 s |
| `PortableDG::LaplaceOperatorDG` (batched) | 18 MiB | 0.78 s |

### Phase 4: polytopal operator on device (open-ended)

Everything in `AgglomerationHandler::reinit*` is host-only by construction (per-call `make_unique` of FEValues, CellId-keyed maps, variable-length quadratures). A device path needs a flattened, setup-time layout:

* per polytope: offsets into a concatenated array of unit quadrature points and weights (the output of `agglomerated_quadrature`), bounding-box extents and origin, DoF offset;
* per polytopal face: offsets into concatenated unit points, weights and normals (what `reinit_master` builds), owner and neighbour polytope ids, the neighbour's bounding box for the second-side evaluation;
* shape functions: because `MappingBox` is a diagonal affine map, values and gradients at arbitrary unit points are cheap: evaluate the 1D Legendre/Lagrange basis per coordinate and multiply. There is no tensor structure in the quadrature, so cost is `n_dofs x n_q` per polytope.

Output can be either an assembled `DeviceCSRMatrix` (atomics into a precomputed sparsity) or a matrix-free polytopal `vmult`. The latter is a research topic rather than an engineering task.

## 5. Memory budget for the Phase 1 fine matrix

Nonzeros per cell for SIP-DG with FE_DGQ(p) in 3D: `(p+1)^3 * (p+1)^3 * 7` (self block plus six neighbours).

| p | dofs/cell | nnz/cell | bytes/cell (double + int32) |
|---|---|---|---|
| 1 | 8 | 448 | 5.4 kB |
| 2 | 27 | 5103 | 61 kB |
| 3 | 64 | 28672 | 344 kB |

At p=2, 1M fine cells need about 61 GB for the fine matrix alone, more than one GPU. This is why Phase 3 matters for 3D and higher order, and why Phase 1 is only a stepping stone there. Phase 3 brings the fine level down to 59 MiB in the measured case.

## 6. Risks and open questions

* `Multigrid`/`PreconditionMG` with `MemorySpace::Default` vectors: verified in `device_agglomerated_mg` (same iteration counts and residuals as the host pipeline at p=1 and p=2 on three ranks).
* Ghost exchange of device vectors uses MPI from device buffers; without GPU-aware MPI deal.II stages through the host. Fine for correctness, matters for performance.
* `Tvmult` via explicit transpose doubles transfer-matrix memory; acceptable because transfers are tall and thin.
* Debug builds: polyDEAL builds a `.g` library. Kokkos bounds checking (`Kokkos_ENABLE_DEBUG_BOUNDS_CHECK`) in deal.II debug builds will slow device tests but catch indexing errors.
* The CI image (`dealii/dealii:master-focal`) has Kokkos Serial, so all device tests run there as host tests. That is the intended fallback.

## 7. Files touched

| File | Change |
|---|---|
| `include/device_sparse_matrix.h` | new: `DeviceCSRMatrix` |
| `include/multigrid_amg.h`, `source/multigrid_amg.cc` | generalise `MGTransferAgglomeration` over matrix type, add device instantiation |
| `include/utils.h` | device `MGCoarseDirect` |
| `include/portable/dg_kokkos_kernels.h` | copied batched kernels (dealiiX-portable-multigrid) |
| `include/portable/laplace_operator_dg.h` | new: `PortableDG::LaplaceOperatorDG`, the matrix-free fine level |
| `include/device_dg_operator.h` | new: point-wise `DeviceDGOperator`, kept for Phase 4 |
| `test/polydeal/device_dg_operator.cc` + `.output` | new test |
| `examples/agglo_mg_device.cc` | new example |
| `test/polydeal/device_csr_vmult.cc`, `device_agglomerated_mg.cc` + `.output` | new tests |
| `README.md` | document CPU/GPU build expectations |
