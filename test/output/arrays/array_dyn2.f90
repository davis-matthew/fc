! RUN: %fc %s -o %t && %t | FileCheck %s
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

!CHECK: 12          13          14          15          16

!CHECK: 13          14          15          16          17

!CHECK: 14          15          16          17          18

!CHECK: 15          16          17          18          19

!CHECK: 16          17          18          19          20

!CHECK: 13          14          15          16          17

!CHECK: 14          15          16          17          18

!CHECK: 15          16          17          18          19

!CHECK: 16          17          18          19          20

!CHECK: 17          18          19          20          21

!CHECK: 14          15          16          17          18

!CHECK: 15          16          17          18          19

!CHECK: 16          17          18          19          20

!CHECK: 17          18          19          20          21

!CHECK: 18          19          20          21          22

!CHECK: 15          16          17          18          19

!CHECK: 16          17          18          19          20

!CHECK: 17          18          19          20          21

!CHECK: 18          19          20          21          22

!CHECK: 19          20          21          22          23

!CHECK: 13          14          15          16          17

!CHECK: 14          15          16          17          18

!CHECK: 15          16          17          18          19

!CHECK: 16          17          18          19          20

!CHECK: 17          18          19          20          21

!CHECK: 14          15          16          17          18

!CHECK: 15          16          17          18          19

!CHECK: 16          17          18          19          20

!CHECK: 17          18          19          20          21

!CHECK: 18          19          20          21          22

!CHECK: 15          16          17          18          19

!CHECK: 16          17          18          19          20

!CHECK: 17          18          19          20          21

!CHECK: 18          19          20          21          22

!CHECK: 19          20          21          22          23

!CHECK: 16          17          18          19          20

!CHECK: 17          18          19          20          21

!CHECK: 18          19          20          21          22

!CHECK: 19          20          21          22          23

!CHECK: 20          21          22          23          24

!CHECK: 14          15          16          17          18

!CHECK: 15          16          17          18          19

!CHECK: 16          17          18          19          20

!CHECK: 17          18          19          20          21

!CHECK: 18          19          20          21          22

!CHECK: 15          16          17          18          19

!CHECK: 16          17          18          19          20

!CHECK: 17          18          19          20          21

!CHECK: 18          19          20          21          22

!CHECK: 19          20          21          22          23

!CHECK: 16          17          18          19          20

!CHECK: 17          18          19          20          21

!CHECK: 18          19          20          21          22

!CHECK: 19          20          21          22          23

!CHECK: 20          21          22          23          24

!CHECK: 17          18          19          20          21

!CHECK: 18          19          20          21          22

!CHECK: 19          20          21          22          23

!CHECK: 20          21          22          23          24

!CHECK: 21          22          23          24          25

!CHECK: 15          16          17          18          19

!CHECK: 16          17          18          19          20

!CHECK: 17          18          19          20          21

!CHECK: 18          19          20          21          22

!CHECK: 19          20          21          22          23

!CHECK: 16          17          18          19          20

!CHECK: 17          18          19          20          21

!CHECK: 18          19          20          21          22

!CHECK: 19          20          21          22          23

!CHECK: 20          21          22          23          24

!CHECK: 17          18          19          20          21

!CHECK: 18          19          20          21          22

!CHECK: 19          20          21          22          23

!CHECK: 20          21          22          23          24

!CHECK: 21          22          23          24          25

!CHECK: 18          19          20          21          22

!CHECK: 19          20          21          22          23

!CHECK: 20          21          22          23          24

!CHECK: 21          22          23          24          25

!CHECK: 22          23          24          25          26
