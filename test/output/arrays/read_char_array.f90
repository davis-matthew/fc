program readtest
  character :: a(5)
  read *, a
  print *, a
end program readtest
! RUN: %fc %s -o %t && %t < ../input/read_char_array.in | FileCheck %s
!CHECK: ftfof
