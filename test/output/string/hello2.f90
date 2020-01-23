program test
      character(len=80) :: string
      character(len=20) :: string2
      string = "hello world"
      string2 = string
      print *, string2, string, string2
end program test
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:  hello world         hello world         hello world        
