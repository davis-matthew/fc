! RUN: %fc %s -o %t && %t | FileCheck %s
program outputdata   

   real, dimension(5) :: x, y(10)  
   integer :: i

   do i = 1, 5
    x(i) = 1.0
   end do

   do i = 1, 10
    y(i) = 10.0
   end do


   print *, x
   print *, y
   
   
end program outputdata

!CHECK: 1.00000000    1.00000000    1.00000000    1.00000000    1.00000000

!CHECK: 10.00000000   10.00000000   10.00000000   10.00000000   10.00000000   10.00000000   10.000
