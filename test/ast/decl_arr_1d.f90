program decl_arr_1d
integer :: g(3:10)
end program decl_arr_1d
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: decl_arr_1d.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() decl_arr_1d
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 decl_arr_1d() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable decl_arr_1d {
!CHECK:     // Symbol List:
!CHECK:     // (2, 12)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_arr_1d
!CHECK:     int32[3:10] g
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK: }
