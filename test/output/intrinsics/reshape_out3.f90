! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  integer :: a(2, 5), r(10)
  integer :: i, j

  do j = 1, 5
    do i = 1, 2
      a(i, j) = j + i
    end do
  end do

  r  = reshape(a, (/10/))
  do i = 1, 10 
    print *, r(i)
  end do
end program test

!CHECK: 2

!CHECK: 3

!CHECK: 3

!CHECK: 4

!CHECK: 4

!CHECK: 5

!CHECK: 5

!CHECK: 6

!CHECK: 6

!CHECK: 7
