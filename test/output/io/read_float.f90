program readtest
  real :: a
  read *, a
  print *, a
end program readtest
! RUN: %fc %s -o %t && %t < ../input/read_float.in | FileCheck %s
!CHECK:  3.456
