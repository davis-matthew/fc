program arr_construct
  integer:: K
  real ::a(4) = (/ 2, (K,K=1,3,1) /)
  print *,a
end program arr_construct
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: array_implied_do1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() arr_construct
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 arr_construct() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable arr_construct {
!CHECK:     // Symbol List:
!CHECK:     // (3, 10)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, arr_construct
!CHECK:     real[1:4] a
!CHECK:     // (2, 13)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, arr_construct
!CHECK:     int32 k
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (3, 10)
!CHECK:   t.2 = (/*IndVar=*/k, /*Init=*/1, /*End=*/3, /*Incr=*/1)
!CHECK:   t.1 = cast (/ 2, t.2 /) to   real
!CHECK:   a = t.1
!CHECK:   // (4, 3)
!CHECK:   printf(a())
!CHECK: }
