program test
logical :: flag1, flag2, flag3, flag4, flag5, flag6
integer :: a = 5
integer :: b = 5
flag1 = a == b 
flag2 = a /= b
flag3 = a < b
flag4 = a <= b
flag5 = a > b
flag6 = a >= b
print *,(a + b)
end program test
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.rel_assignment5.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() test
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 test() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable test {
!CHECK:     // Symbol List: 
!CHECK:     // (3, 12)
!CHECK:     // ID: 8, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 a
!CHECK:     // (4, 12)
!CHECK:     // ID: 9, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 b
!CHECK:     // (2, 12)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     logical flag1
!CHECK:     // (2, 19)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     logical flag2
!CHECK:     // (2, 26)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     logical flag3
!CHECK:     // (2, 33)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     logical flag4
!CHECK:     // (2, 40)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     logical flag5
!CHECK:     // (2, 47)
!CHECK:     // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     logical flag6
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (3, 12)
!CHECK:   a = 5
!CHECK:   // (4, 12)
!CHECK:   b = 5
!CHECK:   // (5, 1)
!CHECK:   t.1 = a == b
!CHECK:   flag1 = t.1
!CHECK:   // (6, 1)
!CHECK:   t.2 = a != b
!CHECK:   flag2 = t.2
!CHECK:   // (7, 1)
!CHECK:   t.3 = a < b
!CHECK:   flag3 = t.3
!CHECK:   // (8, 1)
!CHECK:   t.4 = a <= b
!CHECK:   flag4 = t.4
!CHECK:   // (9, 1)
!CHECK:   t.5 = a > b
!CHECK:   flag5 = t.5
!CHECK:   // (10, 1)
!CHECK:   t.6 = a >= b
!CHECK:   flag6 = t.6
!CHECK:   // (11, 1)
!CHECK:   t.7 = a + b
!CHECK:   printf(t.7)
!CHECK: }
