! RUN: %fc %s -o %t && %t | FileCheck %s
program test
      real :: a, b, c
      a = 10.33542
      b = 10.4535

      c = max(a, b)

      print *, c
end program test

!CHECK: 10.45349979
