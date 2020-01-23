program vinay
  print *, (.true. .and. .false.)
  print *, (.true. .or. .false.)
end program vinay
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:            F
!CHECK:            T
