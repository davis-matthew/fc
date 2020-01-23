program test
      integer :: a = 20
      logical :: flag1 = .true.
      if (flag1) a = 10
      PRINT *,a
end program test
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:           10
