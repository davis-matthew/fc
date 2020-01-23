! RUN: %fc %s -o %t && %t | FileCheck %s
program complex_test
 complex :: a, b, c
 a = (2,3)
 b = (3,5)
 c = a / b
 print *, c
end
!CHECK: (0.61764706, -0.02941176) 
