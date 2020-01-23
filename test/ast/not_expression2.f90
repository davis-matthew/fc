program t
  logical ::a = .TRUE.
  logical :: b = .FALSE.
  if (.not. (a .and. b)) then
    b = .TRUE.
  end if

  print *, b
end program t
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: not_expression2.f90
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
!CHECK:     // (3, 14)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, t
!CHECK:     logical b
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (2, 13)
!CHECK:   a = .TRUE.
!CHECK:   // (3, 14)
!CHECK:   b = .FALSE.
!CHECK:   // (4, 3)
!CHECK:   t.2 = a .AND. b
!CHECK:   t.1 =  .NOT. t.2
!CHECK:   if (t.1) {
!CHECK:     // (5, 5)
!CHECK:     b = .TRUE.
!CHECK:   }
!CHECK:   // (8, 3)
!CHECK:   printf(b)
!CHECK: }
