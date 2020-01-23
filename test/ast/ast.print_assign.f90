program test_print
  integer(kind=4):: a =1 , b =2 , arr(3, 2)
  arr(1,1) = 1
  arr(2, 1) = 1
  arr(3, 1) = 1
  arr(1,2) = 2
  arr(2, 2) = 2
  arr(3, 2) = 2
  print *,arr
end program test_print
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.print_assign.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() test_print
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 test_print() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable test_print {
!CHECK:     // Symbol List: 
!CHECK:     // (2, 21)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test_print
!CHECK:     int32 a
!CHECK:     // (2, 35)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test_print
!CHECK:     int32[1:3, 1:2] arr
!CHECK:     // (2, 28)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test_print
!CHECK:     int32 b
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (2, 21)
!CHECK:   a = 1
!CHECK:   // (2, 28)
!CHECK:   b = 2
!CHECK:   // (3, 3)
!CHECK:   arr(1, 1) = 1
!CHECK:   // (4, 3)
!CHECK:   arr(2, 1) = 1
!CHECK:   // (5, 3)
!CHECK:   arr(3, 1) = 1
!CHECK:   // (6, 3)
!CHECK:   arr(1, 2) = 2
!CHECK:   // (7, 3)
!CHECK:   arr(2, 2) = 2
!CHECK:   // (8, 3)
!CHECK:   arr(3, 2) = 2
!CHECK:   // (9, 3)
!CHECK:   printf(arr())
!CHECK: }
