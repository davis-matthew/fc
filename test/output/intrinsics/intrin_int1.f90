! RUN: %fc %s -o %t && %t | FileCheck %s
program vin
  real:: numbers(10)
  integer:: numbers2(10)
  numbers = 3.24
  print *,INT(numbers)
  numbers2 = INT(numbers)
  print *,numbers2
end program vin

!CHECK: 3            3            3            3            3            3            3           

!CHECK: 3            3            3            3            3            3            3           
