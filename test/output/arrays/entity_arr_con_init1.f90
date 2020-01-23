! RUN: %fc %s -o %t && %t | FileCheck %s
program vin
  real :: numbers(9)=(/  5,  6,  3,  8,  9,  1,  7,  4,  2 /)
  print *, (numbers)
end program vin

!CHECK: 5.00000000    6.00000000    3.00000000    8.00000000    9.00000000    1.00000000    7.0000
