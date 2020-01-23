! this is a comment
program alloctest
  ! this is a comment
  ! this is a comment

  integer, allocatable :: a(:, :) 
  integer :: i, j !this is a comment
  allocate(a(2, 2))

! this is a comment


! this is a comment


  !this is a comment
  do i = 1, 2
    do j = 1, 2
      a(i, j) = i * j
    end do
    !this is a comment
  end do
  print *, a
  !this is a comment
end program alloctest
! this is a comment
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.comment1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (2, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() alloctest
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 alloctest() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable alloctest {
!CHECK:     // Symbol List: 
!CHECK:     // (6, 27)
!CHECK:     // ID: 2, NonConstant, Allocatable, NonTarget, NonPointer, StaticLocal, Intent_None, alloctest
!CHECK:     int32[U, U] a
!CHECK:     // (7, 14)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, alloctest
!CHECK:     int32 i
!CHECK:     // (7, 17)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, alloctest
!CHECK:     int32 j
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (8, 3)
!CHECK:   allocate a[1:2, 1:2]
!CHECK:   // (17, 3)
!CHECK:   t.1 = (/*IndVar=*/i, /*Init=*/1, /*End=*/2, /*Incr=*/1)
!CHECK:   do (t.1) {
!CHECK:     // (18, 5)
!CHECK:     t.2 = (/*IndVar=*/j, /*Init=*/1, /*End=*/2, /*Incr=*/1)
!CHECK:     do (t.2) {
!CHECK:       // (19, 7)
!CHECK:       t.3 = i * j
!CHECK:       a(i, j) = t.3
!CHECK:     }
!CHECK:   }
!CHECK:   // (23, 3)
!CHECK:   printf(a())
!CHECK: }
