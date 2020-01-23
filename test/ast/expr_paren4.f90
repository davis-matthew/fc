program i
  real ::b=10.0
  real  ::c=4.0
  real ::a
  a = (b * c)
end program i
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: expr_paren4.f90
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
!CHECK:     // (4, 10)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     real a
!CHECK:     // (2, 10)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     real b
!CHECK:     // (3, 11)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     real c
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (2, 10)
!CHECK:   b = 10.0
!CHECK:   // (3, 11)
!CHECK:   c = 4.0
!CHECK:   // (5, 3)
!CHECK:   t.1 = b * c
!CHECK:   a = t.1
!CHECK: }
