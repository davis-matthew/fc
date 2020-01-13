subroutine foo() 
end subroutine

program pgm
  integer array(11), array2(11)
  integer :: sum1 = 0
  real c,d
  integer i, j
  c = 10.01

  array(11) = -10;
  do j = 1, 10, 1
      array(j) = j
  end do

  do j = 1, 10, 1
    array2(j) = array(j + 1)
  end do

  print *, array2(10)
end
