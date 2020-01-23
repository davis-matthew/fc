! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  real :: pi1
  real(kind=8) :: pi2


  pi1 = 3.14
  pi2 = 3.1423

  print *, exp(pi1)

  print *, exp(pi2)
end program test

!CHECK: 23.10386848

!CHECK: 23.15706691
