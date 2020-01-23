program test
  integer, parameter :: k = 8
  real (kind = k), parameter :: pi = 3.142
  print *, pi
end program test
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: constprop2.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() test
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 test() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable test {
!CHECK:     // Symbol List:
!CHECK:     // (2, 25)
!CHECK:     // ID: 2, Constant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 k
!CHECK:     // (3, 33)
!CHECK:     // ID: 3, Constant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     double pi
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (2, 25)
!CHECK:   k = 8
!CHECK:   // (3, 33)
!CHECK:   pi = 3.142
!CHECK:   // (4, 3)
!CHECK:   printf(3.142)
!CHECK: }
