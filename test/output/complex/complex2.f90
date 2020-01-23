! RUN: %fc %s -o %t && %t | FileCheck %s
program complex_test
 complex :: a(20)
 a(20) = (2,3)
 a(3) = (2.45,3.0)
 print *, a(20), a(3)
end
!CHECK: (2.00000000, 3.00000000)  (2.45000000, 3.00000000)
