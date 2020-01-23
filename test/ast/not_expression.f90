program t
  logical ::a = .TRUE.
  print *, (.not. a)
end program t
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: not_expression.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() t
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 t() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable t {
!CHECK:     // Symbol List:
!CHECK:     // (2, 13)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, t
!CHECK:     logical a
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (2, 13)
!CHECK:   a = .TRUE.
!CHECK:   // (3, 3)
!CHECK:   t.1 =  .NOT. a
!CHECK:   printf(t.1)
!CHECK: }
