! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  integer :: a(2, 5), r(5, 2)
  integer :: i, j

  do j = 1, 5
    do i = 1, 2
      a(i, j) = j + i
    end do
  end do

  print *, a
  r  = reshape(a, (/5, 2/))
  print *, r
end program test

!CHECK: 2            3            3            4            4            5            5           

!CHECK: 2            3            3            4            4            5            5           
