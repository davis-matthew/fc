! RUN: %fc -emit-ir %s -o - | FileCheck %s
program foo
    !CHECK: omp.parallel
    !$omp parallel
    print *, "hello from openmp world"
    !$omp end parallel 

    print *, "hello from outside"
end program foo
