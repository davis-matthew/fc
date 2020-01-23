! RUN: %fc %s -o %t && %t | FileCheck %s
program t
  integer :: i, j

  L1: do i = 1, 10
    L2: do j = 1, 10 
      if (i + j < 10 ) cycle L1
      print *, i, j
    end do L2 
  end do L1

end program t

!CHECK: 9           1

!CHECK: 9           2

!CHECK: 9           3

!CHECK: 9           4

!CHECK: 9           5

!CHECK: 9           6

!CHECK: 9           7

!CHECK: 9           8

!CHECK: 9           9

!CHECK: 9          10

!CHECK: 10           1

!CHECK: 10           2

!CHECK: 10           3

!CHECK: 10           4

!CHECK: 10           5

!CHECK: 10           6

!CHECK: 10           7

!CHECK: 10           8

!CHECK: 10           9

!CHECK: 10          10
