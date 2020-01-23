! RUN: %fc %s -o %t && %t | FileCheck %s
program matmul_test
  integer, parameter :: n = 300
  integer :: a(n, n), b(n, n), c(n, n)
  integer :: i, j, k

  do j = 1, n
    do i= 1, n
      a(i, j) = i + j
      if ( i == j) then
        b(i, j) = 1
      else
        b(i, j) = 0
      endif
    end do
  end do

  !$omp parallel do
  do i = 1,n
    do j = 1,n
      c(i,j) = 0
      do k = 1,n
        c(i,j) = c(i,j) + a(i,k) * b(k,j)
      enddo
    enddo
  enddo
  !$omp end parallel do

  print *, sum(c)

end
!CHECK: 27090000
