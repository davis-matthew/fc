! RUN: %fc -emit-ast %s -o - | FileCheck %s
program foo
  !$omp parallel
      print *, "Hello from omp"
      !$omp single
        print *, "Hello from single thread"
      !$omp end single
  !$omp end parallel
  print *, "From outsitde"
end program foo
!CHECK: Program: ast.single2.f90
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
!CHECK:   omp parallel {
!CHECK:     // (4, 7)
!CHECK:     printf({Hello from omp})
!CHECK:     // (5, 7)
!CHECK:     omp single {
!CHECK:       // (6, 9)
!CHECK:       printf({Hello from single thread})
!CHECK:     }
!CHECK:   }
!CHECK:   // (9, 3)
!CHECK:   printf({From outsitde})
!CHECK: }
