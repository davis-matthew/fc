program decl_alloca
integer(KIND=8), ALLOCATABLE, DIMENSION(10,20) :: i
end program decl_alloca
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: decl_allocatable.f90
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
!CHECK:     // (2, 51)
!CHECK:     // ID: 2, NonConstant, Allocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_alloca
!CHECK:     int64[1:10, 1:20] i
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK: }
