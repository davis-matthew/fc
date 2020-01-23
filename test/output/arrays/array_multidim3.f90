! RUN: %fc %s -o %t && %t | FileCheck %s
program pgm
integer :: a(3,3,3), i , j, k, c

c = 1
do i = 1, 3
do j = 1, 3
do k = 1, 3
 a(j,i,k) = c
 c = c + 1
enddo
enddo
enddo

do k = 1, 3
do j = 1, 3
do i = 1, 3
 if (a(k,j,i) <= a(i,j,k)) then
  print *, a(k,j,i)
 endif
enddo
enddo
enddo
end

!CHECK: 1

!CHECK: 2

!CHECK: 3

!CHECK: 10

!CHECK: 11

!CHECK: 12

!CHECK: 19

!CHECK: 20

!CHECK: 21

!CHECK: 5

!CHECK: 6

!CHECK: 14

!CHECK: 15

!CHECK: 23

!CHECK: 24

!CHECK: 9

!CHECK: 18

!CHECK: 27
