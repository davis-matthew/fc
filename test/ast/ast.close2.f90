program readtest
  open  (unit = 1, file = '1.txt', status = 'NEW')
  close (unit = 1)
end program readtest
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.close2.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() readtest
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 readtest() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable readtest {
!CHECK:     // Symbol List: 
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (2, 3)
!CHECK:   1 = open({1.txt}, NEW)
!CHECK:   // (3, 3)
!CHECK:   1 = close(/*unit=*/1)
!CHECK: }
