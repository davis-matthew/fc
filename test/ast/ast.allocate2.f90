program alloctest
  integer, allocatable :: a(:, :)
  integer :: i, j

  allocate(a(2, 2))

  do i = 1, 2
    do j = 1, 2
      a(i, j) = i * j
    end do
  end do
  print *, a
end program alloctest
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.allocate2.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() alloctest
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 alloctest() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable alloctest {
!CHECK:     // Symbol List: 
!CHECK:     // (2, 27)
!CHECK:     // ID: 2, NonConstant, Allocatable, NonTarget, NonPointer, StaticLocal, Intent_None, alloctest
!CHECK:     int32[U, U] a
!CHECK:     // (3, 14)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, alloctest
!CHECK:     int32 i
!CHECK:     // (3, 17)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, alloctest
!CHECK:     int32 j
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (5, 3)
!CHECK:   allocate a[1:2, 1:2]
!CHECK:   // (7, 3)
!CHECK:   t.1 = (/*IndVar=*/i, /*Init=*/1, /*End=*/2, /*Incr=*/1)
!CHECK:   do (t.1) {
!CHECK:     // (8, 5)
!CHECK:     t.2 = (/*IndVar=*/j, /*Init=*/1, /*End=*/2, /*Incr=*/1)
!CHECK:     do (t.2) {
!CHECK:       // (9, 7)
!CHECK:       t.3 = i * j
!CHECK:       a(i, j) = t.3
!CHECK:     }
!CHECK:   }
!CHECK:   // (12, 3)
!CHECK:   printf(a())
!CHECK: }
