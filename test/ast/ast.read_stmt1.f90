program readtest
  integer :: a
  read *, a
  print *, a
end program readtest
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.read_stmt1.f90
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
!CHECK:     // (2, 14)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, readtest
!CHECK:     int32 a
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (3, 3)
!CHECK:   read a
!CHECK:   // (4, 3)
!CHECK:   printf(a)
!CHECK: }
