program decl_alloca
  integer ::i,a(10, 10), j
  a = 0
  forall(i=1:10, j=1:10, a(i, j) == 0) a(i, j) = 1
  print *, a
end program decl_alloca
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: for_all1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() decl_alloca
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 decl_alloca() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable decl_alloca {
!CHECK:     // Symbol List:
!CHECK:     // (2, 15)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_alloca
!CHECK:     int32[1:10, 1:10] a
!CHECK:     // (3, 3)
!CHECK:     // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_alloca
!CHECK:     int32 decl_alloca.tmp.0
!CHECK:     // (3, 3)
!CHECK:     // ID: 8, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_alloca
!CHECK:     int32 decl_alloca.tmp.1
!CHECK:     // (2, 13)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_alloca
!CHECK:     int32 i
!CHECK:     // (2, 26)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, decl_alloca
!CHECK:     int32 j
!CHECK:     // (3, 3)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, decl_alloca
!CHECK:     (int32)(...) lbound
!CHECK:     // (3, 3)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, decl_alloca
!CHECK:     (int32)(...) ubound
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (3, 3)
!CHECK:   t.1 = (/*IndVar=*/decl_alloca.tmp.1, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:   do (t.1) {
!CHECK:     // (3, 3)
!CHECK:     t.2 = (/*IndVar=*/decl_alloca.tmp.0, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:     do (t.2) {
!CHECK:       // (3, 3)
!CHECK:       a(decl_alloca.tmp.0, decl_alloca.tmp.1) = 0
!CHECK:     }
!CHECK:   }
!CHECK:   // (4, 3)
!CHECK:   t.3 = (/*IndVar=*/i, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:   do (t.3) {
!CHECK:     // (4, 3)
!CHECK:     t.4 = (/*IndVar=*/j, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:     do (t.4) {
!CHECK:       // (4, 3)
!CHECK:       t.5 = a(i, j) == 0
!CHECK:       if (t.5) {
!CHECK:         // (4, 40)
!CHECK:         a(i, j) = 1
!CHECK:       }
!CHECK:     }
!CHECK:   }
!CHECK:   // (5, 3)
!CHECK:   printf(a())
!CHECK: }
