! RUN: %fc %s -o %t && %t | FileCheck %s
program t
  integer :: i

  do i = 1, 10
    exit
    print *, i
  end do
  print *, "Only this line will be printed"


end program t

!CHECK: Only this line will be printed
