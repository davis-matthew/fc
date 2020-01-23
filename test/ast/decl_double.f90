program decl_double
double precision :: i
end program decl_double
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: decl_double.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() decl_double
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 decl_double() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable decl_double {
!CHECK:     // Symbol List:
!CHECK:     // (2, 21)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_double
!CHECK:     double i
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK: }
