! RUN: %fc %s -o %t && %t | FileCheck %s
program t
  integer :: a = 1
  character(5) array
  array = ' '
  print *, "array =",array
end program t

!CHECK: array =
