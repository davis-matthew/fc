program vinay
  print *, 3
end program vinay
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:            3
