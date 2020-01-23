! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  integer ::  a(4)
  integer:: c,d,e,f
  c = 4
  d = 5
  e = 6
  f = 7
  a = (/ c,d,e,f /)
  print *,a
end program test

!CHECK: 4            5            6            7
