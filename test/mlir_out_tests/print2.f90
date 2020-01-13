program foo
  real    :: a, b, c
  real    :: array(5)
  integer :: i

  a = 1.11
  b = 2.43
  c = 4.56

  print *, a, b, c

  do i = 1, 5
    array(i) = a + 1.1
    a = a + 1.1
  enddo

  print *, array
end program
