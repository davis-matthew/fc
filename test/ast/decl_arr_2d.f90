program decl_arr_2d
real  val(3:10,-4:100)
end program decl_arr_2d
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: decl_arr_2d.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() decl_arr_2d
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 decl_arr_2d() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable decl_arr_2d {
!CHECK:     // Symbol List:
!CHECK:     // (2, 7)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_arr_2d
!CHECK:     real[3:10, -4:100] val
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK: }
