! RUN: %fc %s -o %t && %t | FileCheck %s
program pgm
real c,d
integer i
c = 10.01

if (c > 10.01) then
  c = c + 10.32
else
  c = c - 3.12
endif

print *, c
end

!CHECK: 6.89000034
