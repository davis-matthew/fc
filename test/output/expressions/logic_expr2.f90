program test
      logical :: flag1, flag2
      integer :: a = 5
      integer :: b = 5
      flag1 = a + b == b + 5 .and. .true.
      print *, flag1
      end program test
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK: T
