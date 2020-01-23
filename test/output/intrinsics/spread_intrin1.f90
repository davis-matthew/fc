! RUN: %fc %s -o %t && %t | FileCheck %s
program test

  real :: a(3)
  real :: b(3, 3)

  a(1) = 1.0
  a(2) = 2.0
  a(3) = 4.0

  b = spread(a(1:3), 1, 3)
  print *, b

  b = spread(a(1:3), 2, 3)
  print *, b
end program

!CHECK: 1.00000000    1.00000000    1.00000000    2.00000000    2.00000000    2.00000000    4.0000

!CHECK: 1.00000000    2.00000000    4.00000000    1.00000000    2.00000000    4.00000000    1.0000
