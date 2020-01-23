program vinay
  print *, (3 - (2) ** 2 ** 3)
 end program vinay
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:         -253
