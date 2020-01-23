module vin
integer :: a
end module vin

subroutine pgm
  integer :: a(10)
  a(1) = 10
end subroutine pgm
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: module1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (5, 12)
!CHECK:   // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (void)() pgm
!CHECK:   // (1, 8)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   Module vin
!CHECK: }
!CHECK: Module vin {
!CHECK:   // ModuleScope, Parent: GlobalScope
!CHECK:   SymbolTable vin {
!CHECK:     // Symbol List:
!CHECK:     // (2, 12)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, vin
!CHECK:     int32 a
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:     {
!CHECK:       // (2, 12)
!CHECK:       NAME: a
!CHECK:       SYMBOL ID: 2
!CHECK:     }
!CHECK:   }
!CHECK: }
!CHECK: // Subroutine
!CHECK: void pgm() {
!CHECK:   // SubroutineScope, Parent: GlobalScope
!CHECK:   SymbolTable pgm {
!CHECK:     // Symbol List:
!CHECK:     // (6, 14)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, pgm
!CHECK:     int32[1:10] a
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (7, 3)
!CHECK:   a(1) = 10
!CHECK: }
