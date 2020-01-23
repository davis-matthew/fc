! RUN: %fc %s -o %t && %t | FileCheck %s

program vin
  integer::arr(10,20,30), temp(10)
  arr = 3
  arr(10,20,30) = 30
  temp = reshape(arr(:,20,30),(/ 10 /))
  print *,temp
end program vin

!CHECK: 3            3            3            3            3            3            3           
