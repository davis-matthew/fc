program vinay
  print *, (.not. .false. .and. .not. .false.)
end program vinay
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:            T
