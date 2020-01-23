program decl_dim_arr
integer, DIMENSION(1:10,-1:100, 3:10) :: g
end program decl_dim_arr
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: decl_dimension_arr.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() decl_dim_arr
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 decl_dim_arr() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable decl_dim_arr {
!CHECK:     // Symbol List:
!CHECK:     // (2, 42)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_dim_arr
!CHECK:     int32[1:10, -1:100, 3:10] g
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK: }
