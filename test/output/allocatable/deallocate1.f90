! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  real, dimension(:), allocatable :: array

  allocate(array(10))

  array = 10
  print *, array

  deallocate(array)


end program test

!CHECK: 10.00000000   10.00000000   10.00000000   10.00000000   10.00000000   10.00000000   10.000
