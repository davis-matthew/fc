! RUN: %fc %s -o %t && %t | FileCheck %s
subroutine foo(d)
  real(kind=8) :: d
  print *, d
end subroutine foo

program test
  call foo(0.01d0)
end program

!CHECK: 0.01000000
