! RUN: %fc %s -o %t && %t | FileCheck %s
program vin
  integer :: numbers(9)
  numbers = 3
  numbers(1:3) = numbers(2:4) + numbers(3:5) + 2 - numbers(8)
  print *,numbers
end program vin

!CHECK: 5            5            5            3            3            3            3           
