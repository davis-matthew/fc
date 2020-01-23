! RUN: %fc %s -o %t && %t | FileCheck %s
program vin
  
  integer :: a(10,20+3:25)
  a= 10
  print *,a
end program vin

!CHECK: 10           10           10           10           10           10           10          
