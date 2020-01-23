program i
  integer, dimension(1, 1, 1, 1, 1, 1, 1) :: array
  integer::a
  array(1, 1, 1, 1, 1, 1, 1) = 1 + 1 + 4 ** 2
  a = array(1, 1, 1, 1, 1, 1, 1)

end program i
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: expr_array7d.f90
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
!CHECK:     // (3, 12)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     int32 a
!CHECK:     // (2, 46)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     int32[1:1, 1:1, 1:1, 1:1, 1:1, 1:1, 1:1] array
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (4, 3)
!CHECK:   array(1, 1, 1, 1, 1, 1, 1) = 18
!CHECK:   // (5, 3)
!CHECK:   a = array(1, 1, 1, 1, 1, 1, 1)
!CHECK: }
