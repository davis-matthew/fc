program alloctest
  integer, allocatable :: a(:)
  integer, allocatable :: b(:)
  integer :: i
  integer :: st

  allocate(a(6), b(8), stat = st)
  print *, "Allocation status ", st

  do i = 1, 6
    a(i) = i
  end do

  do i = 1, 10
    b(i) = i
  end do

  print *, a
  print *, b
  deallocate(a, b, stat = st)
  print *, "De Allocation status ", st
end program alloctest
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.deallocate1.f90
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
!CHECK:     int32[U] a
!CHECK:     // (3, 27)
!CHECK:     // ID: 3, NonConstant, Allocatable, NonTarget, NonPointer, StaticLocal, Intent_None, alloctest
!CHECK:     int32[U] b
!CHECK:     // (4, 14)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, alloctest
!CHECK:     int32 i
!CHECK:     // (5, 14)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, alloctest
!CHECK:     int32 st
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (7, 3)
!CHECK:   allocate a[1:6], b[1:8]
!CHECK:   // (8, 3)
!CHECK:   printf({Allocation status }, st)
!CHECK:   // (10, 3)
!CHECK:   t.1 = (/*IndVar=*/i, /*Init=*/1, /*End=*/6, /*Incr=*/1)
!CHECK:   do (t.1) {
!CHECK:     // (11, 5)
!CHECK:     a(i) = i
!CHECK:   }
!CHECK:   // (14, 3)
!CHECK:   t.2 = (/*IndVar=*/i, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:   do (t.2) {
!CHECK:     // (15, 5)
!CHECK:     b(i) = i
!CHECK:   }
!CHECK:   // (18, 3)
!CHECK:   printf(a())
!CHECK:   // (19, 3)
!CHECK:   printf(b())
!CHECK:    // (20, 3)
!CHECK:   deallocate a, b, statst
!CHECK:   // (21, 3)
!CHECK:   printf({De Allocation status }, st)
!CHECK: }
