! RUN: %fc %s -o %t && %t | FileCheck %s
program complex_test
 complex :: a, c
 a = (2,3)
 c = conjg(a)
 print *, c 
end
!CHECK: (2.00000000, -3.00000000)

