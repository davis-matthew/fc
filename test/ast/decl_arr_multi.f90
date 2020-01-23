program decl_arr_multi
real  val1(3:10,4:100, 10)
integer val2(3:10,1003, 10:100)
end program decl_arr_multi
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: decl_arr_multi.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() decl_arr_multi
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 decl_arr_multi() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable decl_arr_multi {
!CHECK:     // Symbol List:
!CHECK:     // (2, 7)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_arr_multi
!CHECK:     real[3:10, 4:100, 1:10] val1
!CHECK:     // (3, 9)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_arr_multi
!CHECK:     int32[3:10, 1:1003, 10:100] val2
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK: }
