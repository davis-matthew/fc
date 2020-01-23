! RUN: %fc -emit-ast %s -o - | FileCheck %s
program foo
  integer  :: n
  n = 10
  !$omp single
      print *, "Hello from omp", n
  !$omp end single

  print *, "From outsitde"
end program foo
!CHECK: Program: ast.single1.f90
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
!CHECK:     // (3, 15)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32 n
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (4, 3)
!CHECK:   n = 10
!CHECK:   // (5, 3)
!CHECK:   omp single {
!CHECK:     // (6, 7)
!CHECK:     printf({Hello from omp}, n)
!CHECK:   }
!CHECK:   // (9, 3)
!CHECK:   printf({From outsitde})
!CHECK: }
