! RUN: %fc %s -o %t && %t | FileCheck %s
program test_sign
    print *, abs(-1.01)
    print *, abs(10)
  end program test_sign

!CHECK: 1.00999999

!CHECK: 10
