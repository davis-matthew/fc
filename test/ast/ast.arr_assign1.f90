program arr1
  integer:: a(2,2)
  a(1,1) = 10
  print *,a(1,1)
end program arr1
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.arr_assign1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() arr1
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 arr1() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable arr1 {
!CHECK:     // Symbol List: 
!CHECK:     // (2, 13)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, arr1
!CHECK:     int32[1:2, 1:2] a
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (3, 3)
!CHECK:   a(1, 1) = 10
!CHECK:   // (4, 3)
!CHECK:   printf(a(1, 1))
!CHECK: }
