program test
      logical :: flag1, flag2, flag3, flag4, flag5, flag6
      integer :: a = 5
      integer :: b = 5
      integer :: array(4, 3)
      array(3, 3) = a
      flag2 = array(3, 3) + a .eq.  a + b

      end program test
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: relational_expr4.f90
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
!CHECK:     // (5, 18)
!CHECK:     // ID: 10, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32[1:4, 1:3] array
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
!CHECK:   // (6, 7)
!CHECK:   array(3, 3) = a
!CHECK:   // (7, 7)
!CHECK:   t.2 = array(3, 3) + a
!CHECK:   t.3 = a + b
!CHECK:   t.1 = t.2 == t.3
!CHECK:   flag2 = t.1
!CHECK: }
