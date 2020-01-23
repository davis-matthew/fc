! RUN: %fc %s -o %t && %t | FileCheck %s
program i
  real(kind=8)::x = 5
  real::y = 10
  print* , cos(0.0)
  print* , sin(3.14)
  print* , log(2.77)
  print* , cos(sin(0.0))
  print* , log(sin(90.0))
  print* , log(cos(x))
  print* , sin(y * cos(log(y)))
end program i

!CHECK: 1.00000000

!CHECK: 0.00159255

!CHECK: 1.01884735

!CHECK: 1.00000000

!CHECK: -0.11205325

!CHECK: -1.25997124

!CHECK: -0.38834009
