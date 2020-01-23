! RUN: %fc %s -emit-llvm -o - | FileCheck %s -check-prefix=LLVM
! RUN: %fc %s -emit-mlir -o - | FileCheck %s -check-prefix=MLIR
program foo
  integer :: i=1, k=10, q = 10, p = 5, j
  integer :: r = 10
  integer :: a(10, 10), b(10, 10), c(10, 10)

  do i = 1, k
    do j = 1, k
    a(i, j) = i + j
    b(i, j) = i * j
    end do
  end do


  !$omp parallel do
  do i = 1, k
    !$omp parallel do
    do j = 1, k
      c(i, j) = a(i, j) + b(i, j)
    end do
    !$omp end parallel do
  end do
  !$omp end parallel do

  print *, c
end program foo
!MLIR: omp.parallel_do
!MLIR: omp.parallel_do
!MLIR: enddo
!MLIR: enddo
!LLVM: @ident
!LLVM: __kmpc_for_static_init_4
!LLVM: __kmpc_fork_call
!LLVM: __kmpc_for_static_fini
!LLVM: __kmpc_for_static_init_4
!LLVM: __kmpc_for_static_fini
!LLVM: __kmpc_fork_call
