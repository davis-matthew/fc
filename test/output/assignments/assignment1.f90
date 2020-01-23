program i
  integer ::a
  a = 3 
  print *,a
end program i
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:            3
