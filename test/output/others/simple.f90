! RUN: %fc %s -o %t && %t | FileCheck %s
program foo
    !$omp parallel
    print *, "hello from openmp world"
    !$omp end parallel

    print *, "hello from outside"
end program foo
!CHECK: {{(^ hello from openmp world$)}}
