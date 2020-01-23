program vin
  integer:: K
  real ::a(4) = (/ 2, (K,K=1,3,1) /)
  print *, (/ 2, (K,K=1,3,2) /)  + (/ (K,K=3,5,2), 6 /)
  do K=1,4
    print *,a(K)
  end do

end program vin
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: array_const_expr2.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() vin
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 vin() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable vin {
!CHECK:     // Symbol List:
!CHECK:     // (3, 10)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, vin
!CHECK:     real[1:4] a
!CHECK:     // (2, 13)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, vin
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
!CHECK:   t.4 = (/*IndVar=*/k, /*Init=*/1, /*End=*/3, /*Incr=*/2)
!CHECK:   t.5 = (/*IndVar=*/k, /*Init=*/3, /*End=*/5, /*Incr=*/2)
!CHECK:   t.3 = (/ 2, t.4 /) + (/ t.5, 6 /)
!CHECK:   printf(t.3)
!CHECK:   // (5, 3)
!CHECK:   t.6 = (/*IndVar=*/k, /*Init=*/1, /*End=*/4, /*Incr=*/1)
!CHECK:   do (t.6) {
!CHECK:     // (6, 5)
!CHECK:     printf(a(k))
!CHECK:   }
!CHECK: }
