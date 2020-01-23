program i
  integer ::b
  integer ::a
  integer ::c
  integer, dimension(5) :: array
  a = 3
  b = 20
  array(1) = 1
  array(2) = 2
  array(3) = 3
  array(4) = array(1) + array(2) * array(3)
  array(5) = a + b
  c = array(1) + array(2) + array(3) + array(4)


end program i
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: expr_array1d.f90
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
!CHECK:     // (5, 28)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     int32[1:5] array
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
!CHECK:   // (6, 3)
!CHECK:   a = 3
!CHECK:   // (7, 3)
!CHECK:   b = 20
!CHECK:   // (8, 3)
!CHECK:   array(1) = 1
!CHECK:   // (9, 3)
!CHECK:   array(2) = 2
!CHECK:   // (10, 3)
!CHECK:   array(3) = 3
!CHECK:   // (11, 3)
!CHECK:   t.2 = array(2) * array(3)
!CHECK:   t.1 = array(1) + t.2
!CHECK:   array(4) = t.1
!CHECK:   // (12, 3)
!CHECK:   t.3 = a + b
!CHECK:   array(5) = t.3
!CHECK:   // (13, 3)
!CHECK:   t.6 = array(1) + array(2)
!CHECK:   t.5 = t.6 + array(3)
!CHECK:   t.4 = t.5 + array(4)
!CHECK:   c = t.4
!CHECK: }
