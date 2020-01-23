program test
      logical :: flag1, flag2, flag3, flag4, flag5, flag6
      integer :: a = 5
      integer :: b = 5
      flag1 = (a .eq. b)
      flag2 = (a .ne. b)
      flag3 = a .lt. b
      flag4 = (a .le. b)
      flag5 = a .gt. b
      flag6 = a .ge. b

      end program test
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: relational_expr2.f90
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
!CHECK:     // (3, 18)
!CHECK:     // ID: 8, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 a
!CHECK:     // (4, 18)
!CHECK:     // ID: 9, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 b
!CHECK:     // (2, 18)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     logical flag1
!CHECK:     // (2, 25)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     logical flag2
!CHECK:     // (2, 32)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     logical flag3
!CHECK:     // (2, 39)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     logical flag4
!CHECK:     // (2, 46)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     logical flag5
!CHECK:     // (2, 53)
!CHECK:     // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     logical flag6
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (3, 18)
!CHECK:   a = 5
!CHECK:   // (4, 18)
!CHECK:   b = 5
!CHECK:   // (5, 7)
!CHECK:   t.1 = a == b
!CHECK:   flag1 = t.1
!CHECK:   // (6, 7)
!CHECK:   t.2 = a != b
!CHECK:   flag2 = t.2
!CHECK:   // (7, 7)
!CHECK:   t.3 = a < b
!CHECK:   flag3 = t.3
!CHECK:   // (8, 7)
!CHECK:   t.4 = a <= b
!CHECK:   flag4 = t.4
!CHECK:   // (9, 7)
!CHECK:   t.5 = a > b
!CHECK:   flag5 = t.5
!CHECK:   // (10, 7)
!CHECK:   t.6 = a >= b
!CHECK:   flag6 = t.6
!CHECK: }
