! RUN: %fc %s -o %t && %t | FileCheck %s
program test_sign
    print *, sign(-12,1)
    print *, sign(-12,0)
    print *, sign(-12,-1)

    print *, sign(-12.,1.)
    print *, sign(-12.,0.)
    print *, sign(-12.,-1.)

  end program test_sign

!CHECK: 12

!CHECK: 12

!CHECK: -12

!CHECK: 12.00000000

!CHECK: 12.00000000

!CHECK: -12.00000000
