! RUN: %fc -emit-ir %s -o - | FileCheck %s
program foo
  integer  :: n
  n = 10
  !CHECK: omp.single
  !$omp single
      print *, "Hello from omp", n
  !$omp end single

  print *, "From outsitde"
end program foo
