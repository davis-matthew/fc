! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  integer :: a, &
    b

  a = &
    5
  b = &
    10
  print *, &
    a * b

end program test

!CHECK: 50
