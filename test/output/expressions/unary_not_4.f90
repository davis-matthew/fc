program vinay
  print *, (.true. .and. .not. .false.)
  print *, (.not. .false.)
  print *, (.not. .true.)
end program vinay
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:            T
!CHECK:            T
!CHECK:            F
