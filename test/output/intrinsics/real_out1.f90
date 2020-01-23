! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  integer, parameter   :: a = 1999

  print *, a
  print *, real(a)
end program test

!CHECK: 1999

!CHECK: 1999.00000000
