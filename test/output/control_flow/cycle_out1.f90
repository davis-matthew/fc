! RUN: %fc %s -o %t && %t | FileCheck %s
program t
  integer :: i

  do i = 1, 10
  if (i == 1) cycle
    print *, i
  end do


end program t

!CHECK: 2

!CHECK: 3

!CHECK: 4

!CHECK: 5

!CHECK: 6

!CHECK: 7

!CHECK: 8

!CHECK: 9

!CHECK: 10
