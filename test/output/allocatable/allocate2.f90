program alloctest
  integer, allocatable :: a(:, :)
  integer :: i, j

  allocate(a(2, 2))

  do i = 1, 2
    do j = 1, 2
      a(i, j) = i * j
    end do
  end do
  print *, a
end program alloctest
! RUN: %fc %s -o %t && %t | FileCheck %s
!CHECK:            1            2            2            4  
