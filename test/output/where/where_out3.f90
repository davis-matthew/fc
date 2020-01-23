! RUN: %fc %s -o %t && %t | FileCheck %s
program t
  integer :: i = 10
  integer, dimension(10) :: a, b

  do i = 1,10,1
    a(i) = i
    b(i) = -i
  end do
  where ( a > 5 )
    a = 10 
    b = 5
  end where

  print *, a
  print *, b

end program t

!CHECK: 1            2            3            4            5           10           10           

!CHECK: -1           -2           -3           -4           -5            5            5          
