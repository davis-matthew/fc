! RUN: %fc -emit-ir %s -o - | FileCheck %s
program foo
  !CHECK: omp.parallel
  !CHECK: omp.single
  !$omp parallel
      print *, "Hello from omp"
      !$omp single
        print *, "Hello from single thread"
      !$omp end single
  !$omp end parallel

  print *, "From outsitde"
end program foo
