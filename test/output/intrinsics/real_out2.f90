! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  real(kind = 8) :: PI = 3.1425

  print *, real(PI)

end program test

!CHECK: 3.14249992
