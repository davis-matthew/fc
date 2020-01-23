! RUN: %fc %s -o %t && %t | FileCheck %s
program alloctest
  integer, allocatable :: a(:, :)
  integer :: i, j

  allocate(a(3, 4))

  do i = 1, 3
    do j = 1, 4

      a(i, j) = i * j
    end do
  end do
  print *, a
end program alloctest

!CHECK: 1            2            3            2            4            6            3           
