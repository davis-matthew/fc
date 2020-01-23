! RUN: %fc %s -o %t && %t | FileCheck %s
program pgm
integer :: a(5:10),i
a(10) = 3

do i = 5,10
 a(i) = i
enddo

do i = 5,10
  print *, a(i)
enddo
end

!CHECK: 5

!CHECK: 6

!CHECK: 7

!CHECK: 8

!CHECK: 9

!CHECK: 10
