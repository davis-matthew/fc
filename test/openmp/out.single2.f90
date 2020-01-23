! RUN: %fc %s -o %t && %t | FileCheck %s
program foo
  !$omp parallel
      print *, "Hello from omp"
      !$omp single
        print *, "Hello from single thread"
      !$omp end single
  !$omp end parallel

  print *, "From outsitde"
end program foo
!CHECK:  Hello from omp
!CHECK:  Hello from single thread
!CHECK:  From outsitde
