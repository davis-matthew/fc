program foo
  integer, pointer :: iptr1, iptr2
  integer, target :: i1, i2

  iptr1 => i1
  iptr2 => i2

  nullify(iptr1, iptr2)
end program foo
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.nullify1.f90
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
!CHECK:     // (3, 22)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, Target, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32 i1
!CHECK:     // (3, 26)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, Target, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32 i2
!CHECK:     // (2, 23)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, Pointer, StaticLocal, Intent_None, foo
!CHECK:     int32* iptr1
!CHECK:     // (2, 30)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, Pointer, StaticLocal, Intent_None, foo
!CHECK:     int32* iptr2
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (5, 3)
!CHECK:   iptr1 => i1
!CHECK:   // (6, 3)
!CHECK:   iptr2 => i2
!CHECK:   // (8, 3)
!CHECK:   nullify(iptr1, iptr2)
!CHECK: }
