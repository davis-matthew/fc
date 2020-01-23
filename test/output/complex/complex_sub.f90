! RUN: %fc %s -o %t && %t | FileCheck %s
program complex_test
 complex :: a, b, c
 a = (2,3)
 b = (3,5)
 c = a - b
 print *, c
end
!CHECK: (-1.00000000, -2.00000000)
