program vinay
  print *,(3  - 2 * (2 - 3 + ( 10 * (-5)) ** 2 + (-100)))  - 1000
end program vinay
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:        -5795
