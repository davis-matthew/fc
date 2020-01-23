! RUN: %fc %s -o %t && %t | FileCheck %s
program foo
    integer (kind = 4) :: int4 = 123_4
    integer (kind = 8) :: int8 = 12343434_8
    real (kind = 4) :: real4 = 3.14_4
    real (kind = 8) :: real8 = 3.14159265_8

    print *, int4, int8
    print *, real4, real8
end program foo

!CHECK: 123    12343434

!CHECK: 3.14000010  3.14159265
