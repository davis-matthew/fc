program readtest
  open  (1, file = '1.txt', status = 'NEW')
  close (1)
  print *, "opend and closed"
end program readtest
!RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK: opend and closed
