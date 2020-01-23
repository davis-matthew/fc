! RUN: %fc %s -o %t && %t | FileCheck %s
program pgm

integer , pointer :: ptr
integer, target :: val

val = 10 
ptr => val
ptr = 30
print *, ptr, val
end


!CHECK: 30          30
