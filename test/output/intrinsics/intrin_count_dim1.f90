! RUN: %fc %s -o %t && %t | FileCheck %s
program vin
 integer :: numbers(5,10), res(10)
 
  numbers = 5
  res = count(numbers /= 0,dim=1)
  print *,count(numbers == 0,dim=2)
  print *,res
end program vin

!CHECK: 0            0            0            0            0

!CHECK: 5            5            5            5            5            5            5           
