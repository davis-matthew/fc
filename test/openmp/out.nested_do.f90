! RUN: %fc %s -o %t && %t | FileCheck %s
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
!CHECK: 3 5 7 9 11 13 15 17 19 21
!CHECK: 5 8 11 14 17 20 23 26 29 32 7
!CHECK: 11 15 19 23 27 31 35 39 43 9 14
!CHECK: 19 24 29 34 39 44 49 54 11 17 23
!CHECK: 29 35 41 47 53 59 65 13 20 27 34
!CHECK: 41 48 55 62 69 76 15 23 31 39 47
!CHECK: 55 63 71 79 87 17 26 35 44 53 62
!CHECK: 71 80 89 98 19 29 39 49 59 69 79
!CHECK: 89 99 109 21 32 43 54 65 76 87 98
!CHECK: 109 120
