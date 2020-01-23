! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  character(len=10), dimension(10) :: array
  integer :: a(3, 3)
  integer :: i=1, j=2, k=3, val = 4, i2 = 5, j2 = 6, val2 = 7

  !write(*,'(/," At", 2(2i2," change to", i2,:,", "))') i, j, val, i2, j2, val2
  do i = 1, 10
    write(*, '(/"Puzzle ", i0)') i 
  end do

end program test 

!CHECK: Puzzle  1

!CHECK: Puzzle  2

!CHECK: Puzzle  3

!CHECK: Puzzle  4

!CHECK: Puzzle  5

!CHECK: Puzzle  6

!CHECK: Puzzle  7

!CHECK: Puzzle  8

!CHECK: Puzzle  9

!CHECK: Puzzle  10
