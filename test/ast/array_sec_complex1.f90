program test
      integer::a(10,20),c
      integer::b(10,20)
      a = 1
      b = 2
      c = 2

      a(b(1,1):b(2,1) + 1:c,c+2 -4*8 + 16*2) = 10
      print *,a
end program test
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: array_sec_complex1.f90
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
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32[1:10, 1:20] b
!CHECK:     // (2, 25)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 c
!CHECK:     // (4, 7)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, test
!CHECK:     (int32)(...) lbound
!CHECK:     // (4, 7)
!CHECK:     // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 test.tmp.0
!CHECK:     // (4, 7)
!CHECK:     // ID: 8, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 test.tmp.1
!CHECK:     // (5, 7)
!CHECK:     // ID: 9, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 test.tmp.2
!CHECK:     // (5, 7)
!CHECK:     // ID: 10, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 test.tmp.3
!CHECK:     // (8, 7)
!CHECK:     // ID: 11, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 test.tmp.4
!CHECK:     // (4, 7)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, test
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
!CHECK:   // (6, 7)
!CHECK:   c = 2
!CHECK:   // (8, 7)
!CHECK:   t.5 = b(2, 1) + 1
!CHECK:   t.6 = (/*IndVar=*/test.tmp.4, /*Init=*/b(1, 1), /*End=*/t.5, /*Incr=*/c)
!CHECK:   do (t.6) {
!CHECK:     // (8, 7)
!CHECK:     t.9 = c + 2
!CHECK:     t.8 = t.9 - 32
!CHECK:     t.7 = t.8 + 32
!CHECK:     a(test.tmp.4, t.7) = 10
!CHECK:   }
!CHECK:   // (9, 7)
!CHECK:   printf(a())
!CHECK: }
