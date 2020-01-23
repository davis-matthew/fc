! RUN: %fc %s -o %t && %t | FileCheck %s
program vin
  integer :: numbers(9)
  numbers = 3
  print *,count(numbers /= 0)
  print *,count(numbers + numbers == 6)
end program vin

!CHECK: 9

!CHECK: 9
