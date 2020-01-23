! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  real(kind = 8), parameter :: PI = 3.14
  print *, sin(3.14)
end program test

!CHECK: 0.00159255
