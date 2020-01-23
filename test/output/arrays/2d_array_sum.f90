! RUN: %fc %s -o %t && %t | FileCheck %s
program test
  integer :: a(10, 10), b(10,10), c(10, 10)
  integer:: i = 1,j =1
  
  do while (i <= 10)
    j = 1
    do while (j <= 10)
      a(j,i) = i
      b(j,i) = j
      j = j + 1
    end do
    i = i + 1
  end do
 
  i = 1
  do while (i <= 10)
    j = 1
    do while (j <= 10)
      c(j,i) = a(j,i) + b(j,i)
      j = j + 1
    end do
    i = i + 1
  end do
  
  print *, c
end program test

!CHECK: 2            3            4            5            6            7            8           
