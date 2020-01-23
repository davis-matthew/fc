program i
  integer ::b
  integer ::a
  a = 3 
  b = 20
  print *, (a + b)
end program i
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:           23
