program i
  integer ::b
  integer ::a
  integer ::c
  a = 3 
  b = 20
  c = a+b*4*2
  print *,c
end program i
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:          163