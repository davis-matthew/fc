program alloctest
  integer, allocatable :: a(:, :, :)
  integer :: i, j, k
  integer :: m, n, o

  read *, m, n, o

  allocate(a(m, n, o))

  do i = 1, m
    do j = 1, n
      do k = 1, o

        a(i, j, k) = i * j * k
      end do
    end do
  end do
  print *, a
end program alloctest
! RUN: %fc %s -o %t && %t < ../input/allocate_in1.in | FileCheck %s
!CHECK:  1            2            2            4            3            6
!CHECK:  2            4            4            8            6           12
!CHECK:  3            6            6           12            9           18
!CHECK:  4            8            8           16           12           24
