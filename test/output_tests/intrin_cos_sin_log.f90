program i
  real(kind=8)::x = 5
  real::y = 10
  print* , cos(0.0)
  print* , sin(3.14)
  print* , log(2.77)
  print* , cos(sin(0.0))
  print* , log(sin(90.0))
  print* , log(cos(x))
  print* , sin(y * cos(log(y)))
end program i
