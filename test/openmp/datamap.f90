program foo
  integer  :: a, b
  a = 10
  b = 20

  !$omp parallel
  print *, "From OpenMP ", a, b
  !$omp end parallel

  print *, "From outsitde ", a, b
end program foo
