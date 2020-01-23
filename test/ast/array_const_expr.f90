program t
  print *, (/ 1,2 /) <= (/ 2, 3 /)
end program t
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: array_const_expr.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() t
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 t() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable t {
!CHECK:     // Symbol List:
!CHECK:     // (2, 12)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, t
!CHECK:     int32[1:2] t.tmp.0
!CHECK:     // (2, 25)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, t
!CHECK:     int32[1:2] t.tmp.1
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (2, 12)
!CHECK:   t.tmp.0(1) = 1
!CHECK:   // (2, 12)
!CHECK:   t.tmp.0(2) = 2
!CHECK:   // (2, 25)
!CHECK:   t.tmp.1(1) = 2
!CHECK:   // (2, 25)
!CHECK:   t.tmp.1(2) = 3
!CHECK:   // (2, 3)
!CHECK:   t.1 = t.tmp.0() <= t.tmp.1()
!CHECK:   printf(t.1)
!CHECK: }
