! RUN: %fc %s -o %t && %t | FileCheck %s
program test
      real :: a
      a = -10.12
      print *, a
      a = abs(a)
      print *, a
end program test

!CHECK: -10.11999989

!CHECK: 10.11999989
