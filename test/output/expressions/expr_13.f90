program vinay
  if (.true.) then
    print *, .true.
  end if
end program vinay
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:            T
