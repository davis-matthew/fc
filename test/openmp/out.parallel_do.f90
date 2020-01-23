! RUN: %fc %s -o %t && %t | FileCheck %s
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
!CHECK: 3  6  9  12  15  18  21  24  27  30
