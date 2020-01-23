! RUN: %fc %s -o %t && %t | FileCheck %s
program t
  integer :: i = 10
  integer, dimension(10) :: a, b

  do i = 1,10,1
    a(i) = i
    b(i) = -i
  end do
  where ( a  < 7 .and. a > 3)
    a = 22
    b = -22
  elsewhere (a >= 7)
    a =  33
    b = -33
  elsewhere
    a = 44
    b = -44
  end where

  print *, a
  print *, b

end program t

!CHECK: 44           44           44           22           22           22           33          

!CHECK: -44          -44          -44          -22          -22          -22          -33         
