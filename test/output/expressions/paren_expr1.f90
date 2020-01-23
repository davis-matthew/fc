! RUN: %fc %s -o %t && %t | FileCheck %s
program t
  integer :: i 

  i = (4 * (4) ** 2)
  print *, i


end program t

!CHECK: 64
