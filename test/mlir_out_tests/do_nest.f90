program pgm
  integer array(10, 10)
  real c,d
  integer i, j
  c = 10.01

  do j = 1, 10, 1
    do i = 1, 10, 1
      array(i, j) = i + j
    end do
  end do

  print *, array(10, 10)
end
