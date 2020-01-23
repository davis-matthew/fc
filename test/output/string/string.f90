program readtest
      character(len=34) :: msg
      read *, msg
      print *, msg
end program readtest
! RUN: %fc %s -o %t && %t < ../input/string.in | FileCheck %s
!CHECK: thisisreallylongmessage! 
