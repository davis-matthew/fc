program vin
  integer:: i
  v1: do i =1,4
    print *,i
    return
  end do v1
end program vin
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: do_label_and_return.f90
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
!CHECK:   // (3, 7)
!CHECK:   t.1 = (/*IndVar=*/i, /*Init=*/1, /*End=*/4, /*Incr=*/1)
!CHECK:   v1: do (t.1) {
!CHECK:     // (4, 5)
!CHECK:     printf(i)
!CHECK:     // (5, 5)
!CHECK:     return
!CHECK:   }
!CHECK: }
