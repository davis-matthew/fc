
subroutine pgm(a)
  integer, intent(in) :: a(10)
  a(1) = 10
end subroutine pgm
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: sub_arg1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (2, 12)
!CHECK:   // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (void)(int32[1:10]) pgm
!CHECK: }
!CHECK: // Subroutine
!CHECK: void pgm(int32[1:10] a) {
!CHECK:   // SubroutineScope, Parent: GlobalScope
!CHECK:   SymbolTable pgm {
!CHECK:     // Symbol List:
!CHECK:     // (2, 12)
!CHECK:     // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Argument, In, pgm
!CHECK:     int32[1:10] a
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (4, 3)
!CHECK:   a(1) = 10
!CHECK: }
