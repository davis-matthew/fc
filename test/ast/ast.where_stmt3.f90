program t
  integer :: i = 10
  integer, dimension(10) :: a, b

  do i = 1,10,1
    a(i) = i
    b(i) = -i
  end do
  where ( a > 3 ) 
    a = 3
    b = 4
  elsewhere
    a = 10
    b = 4
  end where

  print *, a

end program t
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.where_stmt3.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() t
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 t() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable t {
!CHECK:     // Symbol List: 
!CHECK:     // (3, 29)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, t
!CHECK:     int32[1:10] a
!CHECK:     // (3, 32)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, t
!CHECK:     int32[1:10] b
!CHECK:     // (2, 14)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, t
!CHECK:     int32 i
!CHECK:     // (9, 11)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, t
!CHECK:     (int32)(...) lbound
!CHECK:     // (9, 11)
!CHECK:     // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, t
!CHECK:     int32 t.tmp.0
!CHECK:     // (9, 11)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, t
!CHECK:     (int32)(...) ubound
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (2, 14)
!CHECK:   i = 10
!CHECK:   // (5, 3)
!CHECK:   t.1 = (/*IndVar=*/i, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:   do (t.1) {
!CHECK:     // (6, 5)
!CHECK:     a(i) = i
!CHECK:     // (7, 5)
!CHECK:     t.2 = 0 - i
!CHECK:     b(i) = t.2
!CHECK:   }
!CHECK:   // (9, 3)
!CHECK:   t.3 = (/*IndVar=*/t.tmp.0, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:   do (t.3) {
!CHECK:     // (9, 3)
!CHECK:     t.4 = a(t.tmp.0) > 3
!CHECK:     if (t.4) {
!CHECK:       // (10, 5)
!CHECK:       a(t.tmp.0) = 3
!CHECK:       // (11, 5)
!CHECK:       b(t.tmp.0) = 4
!CHECK:     }
!CHECK:     // (9, 3)
!CHECK:     else {
!CHECK:       // (13, 5)
!CHECK:       a(t.tmp.0) = 10
!CHECK:       // (14, 5)
!CHECK:       b(t.tmp.0) = 4
!CHECK:     }
!CHECK:   }
!CHECK:   // (17, 3)
!CHECK:   printf(a())
!CHECK: }
