program i
  integer ::b
  integer ::a
  integer ::c
  a = 3 
  b = 20
  c = a+b
  print *,c
end program i
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.assignment3.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() i
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 i() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable i {
!CHECK:     // Symbol List: 
!CHECK:     // (3, 13)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     int32 a
!CHECK:     // (2, 13)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     int32 b
!CHECK:     // (4, 13)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     int32 c
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (5, 3)
!CHECK:   a = 3
!CHECK:   // (6, 3)
!CHECK:   b = 20
!CHECK:   // (7, 3)
!CHECK:   t.1 = a + b
!CHECK:   c = t.1
!CHECK:   // (8, 3)
!CHECK:   printf(c)
!CHECK: }
