program readtest
  open  (unit = 1, file = '1.txt', status = 'NEW')
  print *, "successfully opened"
  close(1)
end program readtest
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK: successfully opened
