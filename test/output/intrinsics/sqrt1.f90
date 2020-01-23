! RUN: %fc %s -o %t && %t | FileCheck %s
program test
      real :: a
      a = 10.01
      print *, sqrt(a)
end program test

!CHECK: 3.16385841
