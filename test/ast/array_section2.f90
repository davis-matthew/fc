program test
  integer, dimension(10, 10) :: array
  integer, parameter :: l = 1
  integer, parameter :: u = 10
  array(l:u, l:u) = 10
end program test
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: array_section2.f90
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
!CHECK:     // (2, 33)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32[1:10, 1:10] array
!CHECK:     // (3, 25)
!CHECK:     // ID: 3, Constant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 l
!CHECK:     // (5, 3)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 test.tmp.0
!CHECK:     // (5, 3)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 test.tmp.1
!CHECK:     // (4, 25)
!CHECK:     // ID: 4, Constant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 u
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (3, 25)
!CHECK:   l = 1
!CHECK:   // (4, 25)
!CHECK:   u = 10
!CHECK:   // (5, 3)
!CHECK:   t.1 = (/*IndVar=*/test.tmp.1, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:   do (t.1) {
!CHECK:     // (5, 3)
!CHECK:     t.2 = (/*IndVar=*/test.tmp.0, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:     do (t.2) {
!CHECK:       // (5, 3)
!CHECK:       array(test.tmp.0, test.tmp.1) = 10
!CHECK:     }
!CHECK:   }
!CHECK: }
