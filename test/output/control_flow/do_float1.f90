! RUN: %fc %s -o %t && %t | FileCheck %s
program pgm
real c,d
integer i
c = 10.01

do i = 1,10
  c = c + 10.32
end do

print *, c
end

!CHECK: 113.20999908
