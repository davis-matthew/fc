! RUN: %fc -emit-ir %s -o - | FileCheck %s
program foo
  integer :: i, k
  !CHECK: omp.do
  !$omp do
  do i = 1, 10
    print *,  "Hello omp do "
  enddo
  !$omp end do
end program foo
