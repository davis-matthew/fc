! RUN: %fc %s -o %t && %t | FileCheck %s
program pgm
real c,d
integer :: i = 3
c = 10.01

if (c > 10.0) then
  if (i >= 3) c = c + 10.32
else
  c = c - 3.12
endif

print *, c
end

!CHECK: 20.32999992
