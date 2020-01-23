program i
  integer, parameter::a = 4
  integer, parameter::b = 10

  stop (a+b)
  
  
end program i
! RUN: %fc %s -o %t && %t | FileCheck %s
! XFAIL: true
