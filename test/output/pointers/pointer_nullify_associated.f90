! RUN: %fc %s -o %t && %t | FileCheck %s
program nullify_test 
 integer, pointer :: a
 integer, target ::  b, c
 nullify(a)
 print *, associated(a) !CHECK: F
 a => b
 print *, associated(a) ! CHECK : T
 print *, associated(a, b) !CHECK : T
 print *, associated(a, c) !CHECK : F
 nullify(a)
 print *, associated(a) ! CHECK:  F
 print *, associated(a, b) !CHECK: F
 a => c
 print *, associated(a, c) !CHECK: T
end

