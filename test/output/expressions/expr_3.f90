program vinay
  print *, (3 <= 5 .AND. 4 >= 4 .AND. 4 >= 6)
end program vinay
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:            F
