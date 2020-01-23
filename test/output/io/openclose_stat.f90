! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  integer                               :: closestat = -1
  integer                               :: openstat = -1

  print *, "Before ", openstat, closestat
  open(unit=10, file = "1.in", iostat=openstat)
  close(10, iostat=closestat)
  print *, "After ", openstat, closestat

end program test

!CHECK: Before                       -1          -1

!CHECK: After                         0           0
