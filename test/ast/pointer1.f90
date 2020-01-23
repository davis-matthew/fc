program foo
  integer, target :: i
  integer, target :: arr(10)
  integer, pointer :: iptr
  integer, pointer :: arrptr(:)

  iptr => i
  arrptr => arr
end program foo
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: pointer1.f90
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
!CHECK:     // ID: 3, NonConstant, NonAllocatable, Target, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32[1:10] arr
!CHECK:     // (5, 23)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, Pointer, StaticLocal, Intent_None, foo
!CHECK:     int32[U]* arrptr
!CHECK:     // (2, 22)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, Target, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32 i
!CHECK:     // (4, 23)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, Pointer, StaticLocal, Intent_None, foo
!CHECK:     int32* iptr
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (7, 3)
!CHECK:   iptr => i
!CHECK:   // (8, 3)
!CHECK:   arrptr() => arr()
!CHECK: }
