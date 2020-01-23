! RUN: %fc %s -o %t && %t | FileCheck %s
program t
  integer :: i, j, k
  integer, dimension(3, 3, 3) :: a

  do i = 1, 3
    do j = 1, 3
      do k = 1, 3
        a(i, j, k) = i + j + k
      end do
    end do 
  end do

  print *, "Before "
  print *, a

  where ( a(1, :, 1)  > 0 ) &
    a(1, :, 1) = -3

  print *, "After "
  print *, a

end program t

!CHECK: Before

!CHECK: 3            4            5            4            5            6            5           

!CHECK: After

!CHECK: -3            4            5           -3            5            6           -3          
