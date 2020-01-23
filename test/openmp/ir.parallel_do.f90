! RUN: %fc %s -emit-llvm -o - | FileCheck %s -check-prefix=LLVM
! RUN: %fc %s -emit-mlir -o - | FileCheck %s -check-prefix=MLIR
program foo
  integer :: i=1, k=10, q = 10, p = 5
  integer :: r = 10
  integer :: a(10), b(10), c(10)

  do i = 1, k
    a(i) = i
    b(i) = i + i
  end do

  !$omp parallel do
  do i = 1, k
    c(i) = a(i) + b(i)
  end do
  !$omp end parallel do

  print *, c
end program foo
!LLVM: @ident
!LLVM: __kmpc_for_static_init_4
!LLVM: __kmpc_for_static_fini
!LLVM: __kmpc_fork_call
!MLIR: omp.parallel_do
!MLIR: enddo
