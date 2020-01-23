! RUN: %fc %s -o %t && %t | FileCheck %s
program vin
  print *, mod(10,3)
end program vin

!CHECK: 1
