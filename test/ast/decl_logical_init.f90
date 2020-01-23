program decl_log_init
logical:: a, b, c, d = .false. ,f , e = .TRUE.
end program decl_log_init
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: decl_logical_init.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() decl_log_init
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 decl_log_init() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable decl_log_init {
!CHECK:     // Symbol List:
!CHECK:     // (2, 11)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_log_init
!CHECK:     logical a
!CHECK:     // (2, 14)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_log_init
!CHECK:     logical b
!CHECK:     // (2, 17)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_log_init
!CHECK:     logical c
!CHECK:     // (2, 20)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_log_init
!CHECK:     logical d
!CHECK:     // (2, 37)
!CHECK:     // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_log_init
!CHECK:     logical e
!CHECK:     // (2, 33)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_log_init
!CHECK:     logical f
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK: }
