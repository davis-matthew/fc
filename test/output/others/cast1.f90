! RUN: %fc %s -o %t && %t | FileCheck %s
program test
     integer :: i
     real :: r
     integer(kind = 8) :: l
     real(kind = 8) :: d
     i = 10
     r = i
     print *, r
     l = i
     print *, l
     d = r
     print *, d
end program test

!CHECK: 10.00000000

!CHECK: 10

!CHECK: 10.00000000
