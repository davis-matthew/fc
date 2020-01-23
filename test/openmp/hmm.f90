! RUN: %fc %s -o %t && %t | FileCheck %s
program hmm_test
  integer, parameter :: n = 100
  integer :: a(n, n), b(n, n), c(n, n)
  integer :: i, j

  do j = 1, n
    do i= 1, n
      a(i, j) = i + j
      b(i, j) = 2 * (i + j)
    end do
  end do

  !$omp parallel do
  do i = 1,n
    !$omp parallel do
    do j = 1,n
        c(i,j) = a(i,j) * b(i,j)
    enddo
    !$omp end parallel do
  enddo
  !$omp end parallel do

  print *, sum(c)

end
!CHECK: 237350000
