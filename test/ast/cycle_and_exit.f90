program vin
  integer::i
  do i =1,10

  if (i == 5) cycle
  if (i == 9) exit
  print *,i
  end do
end program vin
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: cycle_and_exit.f90
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
!CHECK:     // (2, 12)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, vin
!CHECK:     int32 i
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (3, 3)
!CHECK:   t.1 = (/*IndVar=*/i, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:   do (t.1) {
!CHECK:     // (5, 3)
!CHECK:     t.2 = i == 5
!CHECK:     if (t.2) {
!CHECK:       // (5, 15)
!CHECK:       cycle
!CHECK:     }
!CHECK:     // (6, 3)
!CHECK:     t.3 = i == 9
!CHECK:     if (t.3) {
!CHECK:       // (6, 15)
!CHECK:       exit
!CHECK:     }
!CHECK:     // (7, 3)
!CHECK:     printf(i)
!CHECK:   }
!CHECK: }
