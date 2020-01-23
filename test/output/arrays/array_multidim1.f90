! RUN: %fc %s -o %t && %t | FileCheck %s
program pgm
integer :: a(3,3, 3)
a(1,1, 1) = 2
print *,a(1,1,1)
end

!CHECK: 2
