program test
      character(len=80) :: string
      character(len=80) :: string2
      string = "in this test case we are trying to print really long message!!!!"
      string2 = string
      write(*, *) string2
      write(*, *) string
end program test
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:  in this test case we are trying to print really long message!!!!
!CHECK:  in this test case we are trying to print really long message!!!!
