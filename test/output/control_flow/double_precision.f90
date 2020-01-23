! RUN: %fc %s -o %t && %t | FileCheck %s
program test
      real(kind = 8), parameter :: val1 = 3.14845624E4
      real(kind = 8), parameter :: val2 = 3.14845624E+4
      real(kind = 8), parameter :: val3 = 3.14845624E-4
      print *, val1
      print *, val2
      print *, val3
end program test

!CHECK: 31484.56250000

!CHECK: 31484.56250000

!CHECK: 0.00031485
