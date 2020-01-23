program test
  character (len = 80) :: msg
  character :: ch
  integer :: i
  msg = "hello world"
  i = 3
  ch = msg(i:i)
  print *, ch
end program test
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:            l
