program decl_all
integer a
integer :: b, c = -10
real :: d = -99.34
logical :: e = .true.
real :: arr(10,-1:100), f
integer, DIMENSION(1:10,-1:100, -1:10) :: g

end program decl_all
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: decl_multi.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() decl_all
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 decl_all() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable decl_all {
!CHECK:     // Symbol List:
!CHECK:     // (2, 9)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_all
!CHECK:     int32 a
!CHECK:     // (6, 9)
!CHECK:     // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_all
!CHECK:     real[1:10, -1:100] arr
!CHECK:     // (3, 12)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_all
!CHECK:     int32 b
!CHECK:     // (3, 15)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_all
!CHECK:     int32 c
!CHECK:     // (4, 9)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_all
!CHECK:     real d
!CHECK:     // (5, 12)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_all
!CHECK:     logical e
!CHECK:     // (6, 25)
!CHECK:     // ID: 8, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_all
!CHECK:     real f
!CHECK:     // (7, 43)
!CHECK:     // ID: 9, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_all
!CHECK:     int32[1:10, -1:100, -1:10] g
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK: }
