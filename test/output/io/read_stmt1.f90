program readtest
  integer :: a
  read *, a
  print *, a
end program readtest
! RUN: %fc %s -o %t && %t < ../input/read_stmt1.in | FileCheck %s
!CHECK:          5
