program test
  integer  :: a(10) ,I = 0,j

  a = (/ (I, I = 1, 10) /)
  print *,I
  print *,a
end program test
