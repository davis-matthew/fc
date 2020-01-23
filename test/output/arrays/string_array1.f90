program test
  character(len=10), dimension(5) :: array
  character(len=10) :: message
  character, dimension(10) :: charArray
  integer :: a(5)
  integer :: i, j, k

  do i = 1, 5
    read *, array(i)
  end do 

  do i = 1, 5
    print *, array(i)
  end do 

  do i = 1, 5
    read(array(i), *) a(i)
  end do
  
  print *, a

end program test 
! RUN: %fc %s -o %t && %t < ../input/string_array1.in | FileCheck %s
!CHECK:  1                  
!CHECK:  2                  
!CHECK:  3                  
!CHECK:  4                  
!CHECK:  5                  
!CHECK:            1            2            3            4            5  
