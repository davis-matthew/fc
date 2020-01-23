program vinay
  print *, (3 <= 4) .and. .not. (3 >= 6)
end program vinay
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:            T
