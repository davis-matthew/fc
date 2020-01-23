! RUN: %fc %s -o %t && %t | FileCheck %s
program complex_test
 complex :: a, b, c
 a = (2,3)
 b = (3,5)
 c = a * b
 print *, c
end
!CHECK: (-9.00000000, 19.00000000) 
