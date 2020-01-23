! RUN: %fc %s -o %t && %t | FileCheck %s
program complex_test
 complex :: a, b, c
 a = (2,3)
 b = (3,4)
 c = a + b
 print *, c
end
!CHECK: (5.00000000, 7.00000000)
