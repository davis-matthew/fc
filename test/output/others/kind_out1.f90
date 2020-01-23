! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  integer, parameter :: k = 8
  real (kind = k), parameter :: pi = 3.142
  print *, pi
end program test

!CHECK: 3.14199996
