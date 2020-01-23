! RUN: %fc %s -o %t && %t | FileCheck %s

program t

    real :: r(2,2)
    r = 23.12
    r(1:1,2:2) = 3.12
    r(1:1,1:2) = 4.1
    print *, r

end program t

!CHECK: 4.09999990   23.12000084    4.09999990   23.12000084
