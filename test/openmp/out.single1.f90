! RUN: %fc %s -o %t && %t | FileCheck %s
program foo
  integer  :: n
  n = 10
  !$omp single
      print *, "Hello from omp", n
  !$omp end single

  print *, "From outsitde"
end program foo
!CHECK: Hello from omp    10
!CHECK: From outsitde
