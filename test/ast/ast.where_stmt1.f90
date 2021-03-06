program t
  integer :: i = 10
  integer, dimension(10) :: a

  do i = 1,10,1
    a(i) = i
  end do
  where ( a > 3 ) a = 3

end program t
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.where_stmt1.f90
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
!CHECK:     // (2, 14)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, t
!CHECK:     int32 i
!CHECK:     // (8, 11)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, t
!CHECK:     (int32)(...) lbound
!CHECK:     // (8, 11)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, t
!CHECK:     int32 t.tmp.0
!CHECK:     // (8, 11)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, t
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
!CHECK:   }
!CHECK:   // (8, 3)
!CHECK:   t.2 = (/*IndVar=*/t.tmp.0, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:   do (t.2) {
!CHECK:     // (8, 3)
!CHECK:     t.3 = a(t.tmp.0) > 3
!CHECK:     if (t.3) {
!CHECK:       // (8, 19)
!CHECK:       a(t.tmp.0) = 3
!CHECK:     }
!CHECK:   }
!CHECK: }
