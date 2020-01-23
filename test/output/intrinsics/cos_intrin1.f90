! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  real :: pi = 3.142
  print *, cos(pi)
  print *, cos(0.0)
end program

!CHECK: -0.99999994

!CHECK: 1.00000000
