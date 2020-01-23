! RUN: %fc %s -o %t && %t | FileCheck %s
program t
  integer :: i

  do i = 1, 10
  if (i == 5) exit
    print *, i
  end do


end program t

!CHECK: 1

!CHECK: 2

!CHECK: 3

!CHECK: 4
