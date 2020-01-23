program arr1
  integer:: a(2,2)
  a(1,1) = 10
  print *,a(1,1)
end program arr1
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:           10
