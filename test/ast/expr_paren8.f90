program i
  real ::b=10.0 * (4+4) * ( 4 ** 2)
  real  ::c=(4.0 ** 2)
  real :: a
  a =  (((b + 2)  * (c - 2.2)) / ((b - 2.2) * (c + 2)))
end program i
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: expr_paren8.f90
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
!CHECK:     // (4, 11)
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
!CHECK:   t.3 = cast 8 to   real
!CHECK:   t.2 = 10.0 * t.3
!CHECK:   t.4 = cast 16 to   real
!CHECK:   t.1 = t.2 * t.4
!CHECK:   b = t.1
!CHECK:   // (3, 11)
!CHECK:   c = 16.000000
!CHECK:   // (5, 3)
!CHECK:   t.7 = b + 2
!CHECK:   t.8 = c - 2.2
!CHECK:   t.6 = t.7 * t.8
!CHECK:   t.10 = b - 2.2
!CHECK:   t.11 = c + 2
!CHECK:   t.9 = t.10 * t.11
!CHECK:   t.5 = t.6 / t.9
!CHECK:   a = t.5
!CHECK: }
