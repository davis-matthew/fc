program arr2
  integer:: a(10,10), b(-10:10)
  integer:: c = -3
  integer:: d = 5
  a(d,d) = 10
  b(c) = 10
  print *,(a(d,d) + b(-3))
end program arr2
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.arr_assign2.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() arr2
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 arr2() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable arr2 {
!CHECK:     // Symbol List: 
!CHECK:     // (2, 13)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, arr2
!CHECK:     int32[1:10, 1:10] a
!CHECK:     // (2, 23)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, arr2
!CHECK:     int32[-10:10] b
!CHECK:     // (3, 13)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, arr2
!CHECK:     int32 c
!CHECK:     // (4, 13)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, arr2
!CHECK:     int32 d
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (3, 13)
!CHECK:   c = -3
!CHECK:   // (4, 13)
!CHECK:   d = 5
!CHECK:   // (5, 3)
!CHECK:   a(d, d) = 10
!CHECK:   // (6, 3)
!CHECK:   b(c) = 10
!CHECK:   // (7, 3)
!CHECK:   t.1 = a(d, d) + b(-3)
!CHECK:   printf(t.1)
!CHECK: }
