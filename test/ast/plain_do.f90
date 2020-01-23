program vin
  integer:: i = 0
  do

  i = i + 1
  if (i == 5) cycle
  if (i == 9) exit
  print *,i
  end do
end program vin
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: plain_do.f90
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
!CHECK:     // (2, 13)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, vin
!CHECK:     int32 i
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (2, 13)
!CHECK:   i = 0
!CHECK:   // (3, 3)
!CHECK:   do {
!CHECK:     // (5, 3)
!CHECK:     t.1 = i + 1
!CHECK:     i = t.1
!CHECK:     // (6, 3)
!CHECK:     t.2 = i == 5
!CHECK:     if (t.2) {
!CHECK:       // (6, 15)
!CHECK:       cycle
!CHECK:     }
!CHECK:     // (7, 3)
!CHECK:     t.3 = i == 9
!CHECK:     if (t.3) {
!CHECK:       // (7, 15)
!CHECK:       exit
!CHECK:     }
!CHECK:     // (8, 3)
!CHECK:     printf(i)
!CHECK:   }
!CHECK: }
