! RUN: %fc %s -o %t && %t | FileCheck %s
subroutine sub1(a)
  integer, intent(inout) :: a
  a = 10
end 

program pgm
integer :: a
call sub1(a)
print *, a
end

!CHECK: 10
