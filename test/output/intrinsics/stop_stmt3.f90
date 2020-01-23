program i
   integer, parameter::a= ((3 + 4) * 2) 
   integer, parameter::b= 3 + 6 / 3 * 3 + 2 
   !expected-error@+1 {{STOP 25}}
   stop a + b
end program i
! RUN: %fc %s -o %t && %t | FileCheck %s
! XFAIL: true
