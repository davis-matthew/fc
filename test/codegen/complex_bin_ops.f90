! RUN: %fc -emit-ir %s -o - | FileCheck %s
program complex_test
 complex :: a, b, add, sub, mul, div
 a = (2,3)
 b = (3,5)
 add = a + b ! CHECK: fc.complex_add
 sub = a - b ! CHECK: fc.complex_sub
 mul = a * b ! CHECK: fc.complex_mul
 div = a / b ! CHECK: fc.complex_div
 print *, add, sub, mul, div
end
