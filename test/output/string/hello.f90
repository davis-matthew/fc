program test
      print *, "hello world"
end program test
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:  hello world        
