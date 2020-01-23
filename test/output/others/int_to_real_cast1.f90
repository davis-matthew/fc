! RUN: %fc %s -o %t && %t | FileCheck %s
program p1

  integer :: c= 10, d = 29
  print *, (2.0 + (c+d))
end 

!CHECK: 41.00000000
