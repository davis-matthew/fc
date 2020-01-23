! RUN: %fc %s -o %t && %t | FileCheck %s
program pgm
real c,d
integer :: i = 3
c = 10.01

if (c > 10.0) then
  do while (i <= 10) 
  c = c + 10.32
  i = i + 1
  enddo
else
  c = c - 3.12
endif

print *, c
end

!CHECK: 92.56999969
