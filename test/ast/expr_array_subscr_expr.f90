program i
  integer, dimension(2, 2) :: array
  integer::b = 1 ** 2 / 1
  integer::c = 2 ** 1
  array(b, b) = b * b
  array(b, c) = b * c
  array(c*1, b**1) = b * c
  array(c**1, c) = b * array(b, c)

end program i
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: expr_array_subscr_expr.f90
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
!CHECK:     // (2, 31)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     int32[1:2, 1:2] array
!CHECK:     // (3, 12)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     int32 b
!CHECK:     // (4, 12)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     int32 c
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (3, 12)
!CHECK:   b = 1
!CHECK:   // (4, 12)
!CHECK:   c = 2
!CHECK:   // (5, 3)
!CHECK:   t.1 = b * b
!CHECK:   array(b, b) = t.1
!CHECK:   // (6, 3)
!CHECK:   t.2 = b * c
!CHECK:   array(b, c) = t.2
!CHECK:   // (7, 3)
!CHECK:   t.3 = b * c
!CHECK:   t.4 = c * 1
!CHECK:   t.5 = b ** 1
!CHECK:   array(t.4, t.5) = t.3
!CHECK:   // (8, 3)
!CHECK:   t.6 = b * array(b, c)
!CHECK:   t.7 = c ** 1
!CHECK:   array(t.7, c) = t.6
!CHECK: }
