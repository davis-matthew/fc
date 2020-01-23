! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  real :: f = 1.2
  real(kind=8) :: d = 3.4

  print *, int(f)
  print *, int(d)
end program test

!CHECK: 1

!CHECK: 3
