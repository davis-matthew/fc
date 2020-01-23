program foo
  integer :: arr(3) = (/1, 2, 3/)
  print *,arr
end program foo
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: decl_arr_ctor.f90
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
!CHECK:     // (2, 14)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32[1:3] arr
!CHECK:     // (2, 14)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32 foo.tmp.0
!CHECK:     // (2, 14)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, foo
!CHECK:     (int32)(...) lbound
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (2, 14)
!CHECK:   foo.tmp.0 = 1
!CHECK:   // (2, 14)
!CHECK:   arr(foo.tmp.0) = 1
!CHECK:   // (2, 14)
!CHECK:   t.1 = foo.tmp.0 + 1
!CHECK:   foo.tmp.0 = t.1
!CHECK:   // (2, 14)
!CHECK:   arr(foo.tmp.0) = 2
!CHECK:   // (2, 14)
!CHECK:   t.2 = foo.tmp.0 + 1
!CHECK:   foo.tmp.0 = t.2
!CHECK:   // (2, 14)
!CHECK:   arr(foo.tmp.0) = 3
!CHECK:   // (3, 3)
!CHECK:   printf(arr())
!CHECK: }
