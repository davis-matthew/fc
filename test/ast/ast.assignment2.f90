program i
  integer ::b
  integer ::a
  a = 3 
  b = 20
  print *, (a + b)
end program i
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.assignment2.f90
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
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (4, 3)
!CHECK:   a = 3
!CHECK:   // (5, 3)
!CHECK:   b = 20
!CHECK:   // (6, 3)
!CHECK:   t.1 = a + b
!CHECK:   printf(t.1)
!CHECK: }
