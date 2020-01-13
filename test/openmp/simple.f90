program foo
    !$omp parallel
    print *, "hello from openmp world"
    !$omp end parallel

    print *, "hello from outside"
end program foo
