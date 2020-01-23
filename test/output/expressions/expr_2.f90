program vinay
  print *, (3.0 + 4.0 - 8.2 + 6.666 ** 1)
 end program vinay
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:   5.46600008
