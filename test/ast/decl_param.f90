program decl_param
double precision, PARAMETER :: i = 10.44
end program decl_param
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: decl_param.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() decl_param
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 decl_param() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable decl_param {
!CHECK:     // Symbol List:
!CHECK:     // (2, 32)
!CHECK:     // ID: 2, Constant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_param
!CHECK:     double i
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK: }
