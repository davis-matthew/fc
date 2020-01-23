program readtest
  open  (unit = 1, file = '1.txt', status = 'NEW')
  close (unit = 1)
  print *, "opened and closed"
end program readtest
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK: opened and closed
