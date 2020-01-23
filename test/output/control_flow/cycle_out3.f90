! RUN: %fc %s -o %t && %t | FileCheck %s
program t
  integer :: i, j

  do i = 1, 10
    do j = 1, 10 
      if (i + j < 10 ) cycle
      print *, i, j
    end do 
  end do

end program t

!CHECK: 1           9

!CHECK: 1          10

!CHECK: 2           8

!CHECK: 2           9

!CHECK: 2          10

!CHECK: 3           7

!CHECK: 3           8

!CHECK: 3           9

!CHECK: 3          10

!CHECK: 4           6

!CHECK: 4           7

!CHECK: 4           8

!CHECK: 4           9

!CHECK: 4          10

!CHECK: 5           5

!CHECK: 5           6

!CHECK: 5           7

!CHECK: 5           8

!CHECK: 5           9

!CHECK: 5          10

!CHECK: 6           4

!CHECK: 6           5

!CHECK: 6           6

!CHECK: 6           7

!CHECK: 6           8

!CHECK: 6           9

!CHECK: 6          10

!CHECK: 7           3

!CHECK: 7           4

!CHECK: 7           5

!CHECK: 7           6

!CHECK: 7           7

!CHECK: 7           8

!CHECK: 7           9

!CHECK: 7          10

!CHECK: 8           2

!CHECK: 8           3

!CHECK: 8           4

!CHECK: 8           5

!CHECK: 8           6

!CHECK: 8           7

!CHECK: 8           8

!CHECK: 8           9

!CHECK: 8          10

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
