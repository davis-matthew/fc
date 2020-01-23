program decl_alloca
  integer,save ::i = 10
  i = 1; i = 2
  i= 3;
  i = 4
end program decl_alloca
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: semicolon1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() decl_alloca
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 decl_alloca() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable decl_alloca {
!CHECK:     // Symbol List:
!CHECK:     // (2, 18)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, decl_alloca
!CHECK:     int32 i
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:     {
!CHECK:       // (2, 18)
!CHECK:       NAME: i
!CHECK:       SYMBOL ID: 2
!CHECK:       INIT: 10
!CHECK:     }
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (3, 3)
!CHECK:   i = 1
!CHECK:   // (3, 10)
!CHECK:   i = 2
!CHECK:   // (4, 3)
!CHECK:   i = 3
!CHECK:   // (5, 3)
!CHECK:   i = 4
!CHECK: }
