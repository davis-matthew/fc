program foo
  integer :: i, k
  !$omp do
  do i = 1, 10
    print *,  "Hello omp do "
  enddo
  !$omp end do
end program foo
! RUN: %fc -emit-ast %s -o - | FileCheck %s
!CHECK: Program: ast.ompdo1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() foo
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 foo() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable foo {
!CHECK:     // Symbol List:
!CHECK:     // (2, 14)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32 i
!CHECK:     // (2, 17)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32 k
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (3, 3)
!CHECK:   omp do {
!CHECK:     // (3, 3)
!CHECK:     t.1 = (/*IndVar=*/i, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:     do (t.1) {
!CHECK:       // (5, 5)
!CHECK:       printf({Hello omp do })
!CHECK:     }
!CHECK:   }
!CHECK: }
