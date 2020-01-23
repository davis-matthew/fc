program readtest
  integer, dimension(4) :: a
  read *, a
  print *, a
end program readtest
! RUN: %fc %s -o %t && %t < ../input/read_stmt2.in | FileCheck %s
!CHECK:            1   2  3 4
