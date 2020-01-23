! RUN: %fc %s -o %t && %t | FileCheck %s
program complex_test
 complex :: a
 a = (2,3)
 print *, a
end
!CHECK: (2.00000000, 3.00000000)
