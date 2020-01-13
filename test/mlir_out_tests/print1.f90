program foo
      integer :: a, b, c
      real :: d, e, f
      integer :: i, j
      integer :: array(10)
      a = 1
      b = 2
      c = 3

      print *, a, b, c
      do i = 1, 10
        array(i) = i
      enddo
      print *, array
end program
