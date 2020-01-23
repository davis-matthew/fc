program test
  integer :: val
  parameter(val = 10)
  print *, val

end program test
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: param_stmt.f90
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
!CHECK:     // (2, 14)
!CHECK:     // ID: 2, Constant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 val
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (4, 3)
!CHECK:   printf(10)
!CHECK: }
