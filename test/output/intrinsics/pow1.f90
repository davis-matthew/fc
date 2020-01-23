! RUN: %fc %s -o %t && %t | FileCheck %s
program foo
  real :: r4_res, r4_x = 2, r4_y = 10
  real(kind = 8) :: r8_res, r8_x = 2.5, r8_y = 3

  integer :: i4_res, i4_x = 3, i4_y = 12

  r4_res = r4_x ** r4_y
  r8_res = r8_x ** r8_y
  print *, r4_res
  print *, r8_res

  r4_res = r4_x ** i4_y
  print *, r4_res
end program foo

!CHECK: 1024.00000000

!CHECK: 15.62500000

!CHECK: 4096.00000000
