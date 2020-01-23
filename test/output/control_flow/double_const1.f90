! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  double precision     :: val
  val = 1d-7
  print *, val
  val = 1d+7
  print *, val
  val = 1d7
  print *, val
  val = 1d0
  print *, val
end program test

!CHECK: 0.00000010

!CHECK: 10000000.00000000

!CHECK: 10000000.00000000

!CHECK: 1.00000000
