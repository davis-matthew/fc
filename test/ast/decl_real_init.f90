program decl_real_init
real :: a, b, c, d = 11.333 ,f , e = 10.01
end program decl_real_init
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: decl_real_init.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() decl_real_init
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 decl_real_init() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable decl_real_init {
!CHECK:     // Symbol List:
!CHECK:     // (2, 9)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_real_init
!CHECK:     real a
!CHECK:     // (2, 12)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_real_init
!CHECK:     real b
!CHECK:     // (2, 15)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_real_init
!CHECK:     real c
!CHECK:     // (2, 18)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_real_init
!CHECK:     real d
!CHECK:     // (2, 34)
!CHECK:     // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_real_init
!CHECK:     real e
!CHECK:     // (2, 30)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_real_init
!CHECK:     real f
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK: }
