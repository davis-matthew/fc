! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  integer, dimension(10) :: array
  integer :: k
  array(1:10) = (/ (k,k=1,10) /)
  print *,array
end program test

!CHECK: 1            2            3            4            5            6            7           
