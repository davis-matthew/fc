program decl_arr
integer :: g(10)
end program decl_arr
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: decl_arr_upper.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() decl_arr
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 decl_arr() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable decl_arr {
!CHECK:     // Symbol List:
!CHECK:     // (2, 12)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_arr
!CHECK:     int32[1:10] g
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK: }
