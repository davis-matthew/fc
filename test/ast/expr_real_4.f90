program foo
    real (kind = 4) :: x = 10.1_4
end program foo
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: expr_real_4.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() foo
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 foo() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable foo {
!CHECK:     // Symbol List:
!CHECK:     // (2, 24)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     real x
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK: }
