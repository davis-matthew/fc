! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  integer                               :: deallocstat = -1
  integer, dimension(:, :), allocatable :: array

  print *, "Before ", deallocstat

  allocate(array(5, 5))
  deallocate(array, stat=deallocstat)

  print *, "After ", deallocstat

end program test

!CHECK: Before                       -1

!CHECK: After                         0
