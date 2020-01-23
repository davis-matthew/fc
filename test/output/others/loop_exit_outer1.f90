! RUN: %fc %s -o %t && %t | FileCheck %s
program arr1
  integer :: i ,j
outer:  do i =1,10
    do j =1,10
     if (j == 6) exit outer
    end do 
  end do outer
  print *,"inner indvar usage", j
  
end program arr1

!CHECK: inner indvar usage            6
