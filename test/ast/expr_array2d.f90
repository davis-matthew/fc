program i
  integer, dimension(2, 2) :: array
  array(1, 1) = 1 + 1
  array(1, 2) = 1 * 2
  array(2, 1) = 1 ** 2
  array(2, 2) = array(1, 1) + array(1, 2) * array(2, 1)

end program i
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: expr_array2d.f90
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
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (3, 3)
!CHECK:   array(1, 1) = 2
!CHECK:   // (4, 3)
!CHECK:   array(1, 2) = 2
!CHECK:   // (5, 3)
!CHECK:   array(2, 1) = 1
!CHECK:   // (6, 3)
!CHECK:   t.2 = array(1, 2) * array(2, 1)
!CHECK:   t.1 = array(1, 1) + t.2
!CHECK:   array(2, 2) = t.1
!CHECK: }
