! RUN: %fc %s -o %t && %t | FileCheck %s
program pgm
integer :: a(10)
a(10) = 3
print *, a(10)
end

!CHECK: 3
