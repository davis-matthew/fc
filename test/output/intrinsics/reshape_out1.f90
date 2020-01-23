! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  integer :: a(3), r(3)
  integer :: i, j

  do j = 1, 3
    !do i = 1, 3
      a(j) = j
    !end do
  end do

  print *, a
  r = reshape(a, (/3/))
  print *, r
end program test

!CHECK: 1            2            3

!CHECK: 1            2            3
