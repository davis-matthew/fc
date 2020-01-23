program v
integer(kind=8) ::d = 9999999999_8
end program v
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: expr_int_8.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() v
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 v() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable v {
!CHECK:     // Symbol List:
!CHECK:     // (2, 19)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, v
!CHECK:     int64 d
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK: }
