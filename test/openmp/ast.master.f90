! RUN: %fc -emit-ast %s -o - | FileCheck %s
program foo
  !$omp master
      print *, "Hello from master"
  !$omp end master

  print *, "From outsitde"
end program foo
!CHECK: Program: ast.master.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (2, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() foo
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 foo() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable foo {
!CHECK:     // Symbol List:
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (3, 3)
!CHECK:   omp master {
!CHECK:     // (4, 7)
!CHECK:     printf({Hello from master})
!CHECK:   }
!CHECK:   // (7, 3)
!CHECK:   printf({From outsitde})
!CHECK: }
