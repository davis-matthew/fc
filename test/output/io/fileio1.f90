program readtest
  integer, dimension(10) :: x
  integer, dimension(10) :: y
  integer :: i


   open  (1, file = '../input/1.dat', status = 'old')
   do i = 1,10
      read(1,*) x(i), y(i)
   end do

   do i = 1,10
      print *, x(i), y(i)
   end do

end program readtest
!RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK: 0 10
!CHECK: 1 11
!CHECK: 2 12
!CHECK: 3 13
!CHECK: 4 14
!CHECK: 5 15
!CHECK: 6 16
!CHECK: 7 17
!CHECK: 8 18
!CHECK: 9 19
