! RUN: %fc %s -o %t && %t | FileCheck %s
program foo
  !$omp master
      print *, "Hello from master"
  !$omp end master

  print *, "From outsitde"
end program foo
!CHECK: Hello from master
!CHECK: From outsitde
