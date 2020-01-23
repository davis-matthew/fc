program vinay
  print *, .not. (3 >= 6)
end program vinay
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:            T
