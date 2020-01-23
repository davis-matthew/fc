program test
      integer::a(10,20)
      integer::b(10,20)
      a = 1
      b = 2
      a(1:5:2,1) = b(1:5:2,2) + b(3:7:2,11)
      PRINT *,a (:,1)
end program test
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: array_sec_operations1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() test
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 test() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable test {
!CHECK:     // Symbol List:
!CHECK:     // (2, 16)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32[1:10, 1:20] a
!CHECK:     // (3, 16)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32[1:10, 1:20] b
!CHECK:     // (4, 7)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, test
!CHECK:     (int32)(...) lbound
!CHECK:     // (4, 7)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 test.tmp.0
!CHECK:     // (4, 7)
!CHECK:     // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 test.tmp.1
!CHECK:     // (5, 7)
!CHECK:     // ID: 8, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 test.tmp.2
!CHECK:     // (5, 7)
!CHECK:     // ID: 9, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 test.tmp.3
!CHECK:     // (6, 7)
!CHECK:     // ID: 10, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 test.tmp.4
!CHECK:     // (6, 20)
!CHECK:     // ID: 11, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 test.tmp.5
!CHECK:     // (6, 33)
!CHECK:     // ID: 12, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 test.tmp.6
!CHECK:     // (4, 7)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, test
!CHECK:     (int32)(...) ubound
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (4, 7)
!CHECK:   t.1 = (/*IndVar=*/test.tmp.1, /*Init=*/1, /*End=*/20, /*Incr=*/1)
!CHECK:   do (t.1) {
!CHECK:     // (4, 7)
!CHECK:     t.2 = (/*IndVar=*/test.tmp.0, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:     do (t.2) {
!CHECK:       // (4, 7)
!CHECK:       a(test.tmp.0, test.tmp.1) = 1
!CHECK:     }
!CHECK:   }
!CHECK:   // (5, 7)
!CHECK:   t.3 = (/*IndVar=*/test.tmp.3, /*Init=*/1, /*End=*/20, /*Incr=*/1)
!CHECK:   do (t.3) {
!CHECK:     // (5, 7)
!CHECK:     t.4 = (/*IndVar=*/test.tmp.2, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:     do (t.4) {
!CHECK:       // (5, 7)
!CHECK:       b(test.tmp.2, test.tmp.3) = 2
!CHECK:     }
!CHECK:   }
!CHECK:   // (6, 33)
!CHECK:   test.tmp.6 = 3
!CHECK:   // (6, 20)
!CHECK:   test.tmp.5 = 1
!CHECK:   // (6, 7)
!CHECK:   t.5 = (/*IndVar=*/test.tmp.4, /*Init=*/1, /*End=*/5, /*Incr=*/2)
!CHECK:   do (t.5) {
!CHECK:     // (6, 7)
!CHECK:     t.6 = b(test.tmp.5, 2) + b(test.tmp.6, 11)
!CHECK:     a(test.tmp.4, 1) = t.6
!CHECK:     // (6, 33)
!CHECK:     t.7 = test.tmp.6 + 2
!CHECK:     test.tmp.6 = t.7
!CHECK:     // (6, 20)
!CHECK:     t.8 = test.tmp.5 + 2
!CHECK:     test.tmp.5 = t.8
!CHECK:   }
!CHECK:   // (7, 7)
!CHECK:   printf(a(:, 1))
!CHECK: }
