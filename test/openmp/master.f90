! RUN: %fc -emit-ir %s -o - | FileCheck %s
program foo
  !CHECK: omp.master
  !$omp master
      print *, "Hello from master"
  !$omp end master

  print *, "From outsitde"
end program foo
