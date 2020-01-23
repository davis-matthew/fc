program t
  integer ::a = 10, b = 20
  print *,(-a + (-b * a))
end program t
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: unary_minus2.f90
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
!CHECK:     // (2, 13)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, t
!CHECK:     int32 a
!CHECK:     // (2, 21)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, t
!CHECK:     int32 b
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (2, 13)
!CHECK:   a = 10
!CHECK:   // (2, 21)
!CHECK:   b = 20
!CHECK:   // (3, 3)
!CHECK:   t.2 = 0 - a
!CHECK:   t.4 = 0 - b
!CHECK:   t.3 = t.4 * a
!CHECK:   t.1 = t.2 + t.3
!CHECK:   printf(t.1)
!CHECK: }
