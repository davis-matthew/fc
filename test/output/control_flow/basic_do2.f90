! RUN: %fc %s -o %t && %t | FileCheck %s
program t
  integer :: i = 10

  do i = 10,1,-1
    print *,i
  end do

end program t

!CHECK: 10

!CHECK: 9

!CHECK: 8

!CHECK: 7

!CHECK: 6

!CHECK: 5

!CHECK: 4

!CHECK: 3

!CHECK: 2

!CHECK: 1
