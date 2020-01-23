program multi_decl
logical :: a, b, c, d = .false. ,f , e = .TRUE.
integer :: g = 10
end program multi_decl
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: multi_decl.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() multi_decl
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 multi_decl() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable multi_decl {
!CHECK:     // Symbol List:
!CHECK:     // (2, 12)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, multi_decl
!CHECK:     logical a
!CHECK:     // (2, 15)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, multi_decl
!CHECK:     logical b
!CHECK:     // (2, 18)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, multi_decl
!CHECK:     logical c
!CHECK:     // (2, 21)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, multi_decl
!CHECK:     logical d
!CHECK:     // (2, 38)
!CHECK:     // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, multi_decl
!CHECK:     logical e
!CHECK:     // (2, 34)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, multi_decl
!CHECK:     logical f
!CHECK:     // (3, 12)
!CHECK:     // ID: 8, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, multi_decl
!CHECK:     int32 g
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK: }
