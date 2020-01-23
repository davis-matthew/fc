! RUN: %fc %s -o %t && %t | FileCheck %s
program pgm
integer :: a(3,3), i , j, c

c = 1
do i = 1, 3
do j = 1, 3
 a(j,i) = c
 c = c + 1
enddo
enddo

do j = 1, 3
do i = 1, 3
 print *, a(j,i)
enddo
enddo
end

!CHECK: 1

!CHECK: 4

!CHECK: 7

!CHECK: 2

!CHECK: 5

!CHECK: 8

!CHECK: 3

!CHECK: 6

!CHECK: 9
