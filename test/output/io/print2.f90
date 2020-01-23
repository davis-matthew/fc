! RUN: %fc %s -o %t && %t | FileCheck %s
program foo
  real    :: a, b, c
  real    :: array(5)
  integer :: i

  a = 1.11
  b = 2.43
  c = 4.56

  print *, a, b, c

  do i = 1, 5
    array(i) = a + 1.1
    a = a + 1.1
  enddo

  print *, array
end program

!CHECK: 1.11000001  2.43000007  4.55999994

!CHECK: 2.21000004    3.30999994    4.40999985    5.50999975    6.60999966
