! RUN: %fc %s -o %t && %t | FileCheck %s

program test

  integer, dimension(:, :), allocatable :: array

  allocate(array(10, 10))
  print *, size(array)
  print *, size(array, 1)
  print *, size(array, 2)
  deallocate(array)

end program test

!CHECK: 100

!CHECK: 10

!CHECK: 10
