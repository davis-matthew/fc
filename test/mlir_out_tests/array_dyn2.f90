subroutine calc(n, m, o)
  integer :: i, j, k
  integer :: m, n, o
  integer :: array(5, 2:n, 3:m, 4:o)
  integer :: t1, t2, t3, t4, t5

  t1 = 3
  t2 = 4
  t3 = 5
  t4 = 6
  t5 = 7

  do k = 4, o
  do j = 3, m
  do i = 2, n
    array(1, i, j, k) = 0
    array(2, i, j, k) = 0
    array(3, i, j, k) = 0
    array(4, i, j, k) = 0
    array(5, i, j, k) = 0
  end do
  end do
  end do

  do k = 4, o
  do j = 3, m
  do i = 2, n
    array(1, i, j, k) = t1 + i + j + k
    array(2, i, j, k) = t2 + i + j + k
    array(3, i, j, k) = t3 + i + j + k
    array(4, i, j, k) = t4 + i + j + k
    array(5, i, j, k) = t5 + i + j + k
  end do
  end do
  end do

  do k = 4, o
  do j = 3, m
  do i = 2, n
    print *, array(1, i, j, k), array(2, i, j, k), array(3, i, j, k), array(4, i, j, k), array(5, i, j, k)
  end do
  end do
  end do
end subroutine

program print_test
  integer :: i, j, k
  integer :: t1, t2, t3, t4, t5
  call calc(6, 6, 7)

end program print_test
