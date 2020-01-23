! RUN: %fc %s -o %t && %t | FileCheck %s
subroutine pgm(n,m)
integer:: n,m, i, j, c
integer :: a(n,m)

c = 0
do i = 1,m
do j = 1,n
  a(j,i) = c
  c = c  + 1
enddo
enddo

do i = 1,m
do j = 1,n
  print *, a(j,i)
enddo
enddo

end

program p

integer :: n = 5 , m = 5
call pgm(n,m)

end

!CHECK: 0

!CHECK: 1

!CHECK: 2

!CHECK: 3

!CHECK: 4

!CHECK: 5

!CHECK: 6

!CHECK: 7

!CHECK: 8

!CHECK: 9

!CHECK: 10

!CHECK: 11

!CHECK: 12

!CHECK: 13

!CHECK: 14

!CHECK: 15

!CHECK: 16

!CHECK: 17

!CHECK: 18

!CHECK: 19

!CHECK: 20

!CHECK: 21

!CHECK: 22

!CHECK: 23

!CHECK: 24
