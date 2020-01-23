! RUN: %fc %s -o %t && %t | FileCheck %s
program foo
  integer :: i, k
  !$omp do
  do i = 1, 10
    print *,  "Hello omp do "
  enddo
  !$omp end do
end program foo
!CHECK:  Hello omp do
!CHECK:  Hello omp do
!CHECK:  Hello omp do
!CHECK:  Hello omp do
!CHECK:  Hello omp do
!CHECK:  Hello omp do
!CHECK:  Hello omp do
!CHECK:  Hello omp do
!CHECK:  Hello omp do
!CHECK:  Hello omp do
