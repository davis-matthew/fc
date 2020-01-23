! RUN: %fc %s -o %t && %t | FileCheck %s
program vin
  integer :: val3(3)
  val3 = (/ 1, 2, 3 /) + (/4,5,6 /) 
  print *,val3
end program vin

!CHECK: 5            7            9
