! RUN: %fc %s -o %t && %t | FileCheck %s
subroutine subr1(arr1, arr2)
  integer :: arr1(*)
  integer, dimension(*) :: arr2
  integer :: i

  do i = 1, 5
    print *, arr1(i), arr2(i)
  end do
end subroutine subr1

program foo
  integer :: arr1(5) = (/1, 2, 3, 4, 5/)
  call subr1(arr1, arr1)
end program foo

!CHECK: 1           1

!CHECK: 2           2

!CHECK: 3           3

!CHECK: 4           4

!CHECK: 5           5
