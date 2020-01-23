! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  integer :: val
  parameter(val = 10)
  print *, val

end program test 

!CHECK: 10
