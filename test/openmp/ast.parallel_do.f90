! RUN: %fc %s -emit-ast -o - | FileCheck %s
program foo
  integer :: i=1, k=10, q = 10, p = 5
  integer :: r = 10
  integer :: a(10), b(10), c(10)

  do i = 1, k
    a(i) = i
    b(i) = i + i
  end do

  !$omp parallel do
  do i = 1, k
    c(i) = a(i) + b(i)
  end do
  !$omp end parallel do

  print *, c
end program foo
!CHECK: Program: ast.parallel_do.f90
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
!CHECK:     // (5, 14)
!CHECK:     // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32[1:10] a
!CHECK:     // (5, 21)
!CHECK:     // ID: 8, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32[1:10] b
!CHECK:     // (5, 28)
!CHECK:     // ID: 9, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32[1:10] c
!CHECK:     // (3, 14)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32 i
!CHECK:     // (3, 19)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32 k
!CHECK:     // (3, 33)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32 p
!CHECK:     // (3, 25)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32 q
!CHECK:     // (4, 14)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32 r
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (3, 14)
!CHECK:   i = 1
!CHECK:   // (3, 19)
!CHECK:   k = 10
!CHECK:   // (3, 25)
!CHECK:   q = 10
!CHECK:   // (3, 33)
!CHECK:   p = 5
!CHECK:   // (4, 14)
!CHECK:   r = 10
!CHECK:   // (7, 3)
!CHECK:   t.1 = (/*IndVar=*/i, /*Init=*/1, /*End=*/k, /*Incr=*/1)
!CHECK:   do (t.1) {
!CHECK:     // (8, 5)
!CHECK:     a(i) = i
!CHECK:     // (9, 5)
!CHECK:     t.2 = i + i
!CHECK:     b(i) = t.2
!CHECK:   }
!CHECK:   // (12, 3)
!CHECK:   omp parallel do {
!CHECK:     // (12, 3)
!CHECK:     t.3 = (/*IndVar=*/i, /*Init=*/1, /*End=*/k, /*Incr=*/1)
!CHECK:     do (t.3) {
!CHECK:       // (14, 5)
!CHECK:       t.4 = a(i) + b(i)
!CHECK:       c(i) = t.4
!CHECK:     }
!CHECK:   }
!CHECK:   // (18, 3)
!CHECK:   printf(c())
!CHECK: }
