program vinay
  integer :: a = 100, b = 20

  print *, a / (-b)
end program vinay
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:           -5
