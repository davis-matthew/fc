program test
      logical :: flag1, flag2
      integer :: a = 5
      integer :: b = 5
      flag1 = (a .eq. b) .and. (b .eq. a)
      flag2 = a == b .or. b == a
      end program test
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: logical_expr1.f90
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
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 a
!CHECK:     // (4, 18)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 b
!CHECK:     // (2, 18)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     logical flag1
!CHECK:     // (2, 25)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     logical flag2
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
!CHECK:   t.2 = a == b
!CHECK:   t.3 = b == a
!CHECK:   t.1 = t.2 .AND. t.3
!CHECK:   flag1 = t.1
!CHECK:   // (6, 7)
!CHECK:   t.5 = a == b
!CHECK:   t.6 = b == a
!CHECK:   t.4 = t.5 .OR. t.6
!CHECK:   flag2 = t.4
!CHECK: }
