! RUN: %fc %s -o %t && %t | FileCheck %s
program vin
  integer:: val(10,5)
  val = 5

  print *, all(val == 5)
  print *, all(val /= 5)
  print *,all(val == 5, dim=2)
  print *,all(val == 5, dim=1)
  print *,all(val /= 5, dim=2)
  print *,all(val /= 5, dim=1)
end program vin

!CHECK: T

!CHECK: F

!CHECK: T            T            T            T            T            T            T           

!CHECK: T            T            T            T            T

!CHECK: F            F            F            F            F            F            F           

!CHECK: F            F            F            F            F
