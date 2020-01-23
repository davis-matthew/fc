program p

integer::a=10,d=1

call sub1
call sub2()

contains

subroutine sub1
  integer :: b = 20
  print *,b
  print *,d
  a = 20
end subroutine sub1

subroutine sub2
  integer :: b = 30
  print *,b
  print *,a
end subroutine sub2
end program  p
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: nested_subroutines1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() p
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 p() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable p {
!CHECK:     // Symbol List:
!CHECK:     // (3, 10)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, p
!CHECK:     int32 a
!CHECK:     // (3, 15)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, p
!CHECK:     int32 d
!CHECK:     // (5, 6)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, p
!CHECK:     (void)() sub1
!CHECK:     // (6, 6)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, p
!CHECK:     (void)() sub2
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (3, 10)
!CHECK:   a = 10
!CHECK:   // (3, 15)
!CHECK:   d = 1
!CHECK:   // (5, 6)
!CHECK:   call sub1()
!CHECK:   // (6, 6)
!CHECK:   call sub2()
!CHECK:   // Internal SubProgram Lists:
!CHECK:   // Subroutine
!CHECK:   void sub1() {
!CHECK:     // SubroutineScope, Parent: MainProgramScope
!CHECK:     SymbolTable sub1 {
!CHECK:       // Symbol List:
!CHECK:       // (14, 3)
!CHECK:       // ID: 8, ParentSymbol: 2, ParentSymbolTable: p
!CHECK:       int32 a
!CHECK:       // (11, 14)
!CHECK:       // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, sub1
!CHECK:       int32 b
!CHECK:       // (13, 11)
!CHECK:       // ID: 7, ParentSymbol: 3, ParentSymbolTable: p
!CHECK:       int32 d
!CHECK:     }
!CHECK:     // Specification Constructs:
!CHECK:     EntityDeclList {
!CHECK:       {
!CHECK:         // (11, 14)
!CHECK:         NAME: b
!CHECK:         SYMBOL ID: 6
!CHECK:         INIT: 20
!CHECK:       }
!CHECK:     }
!CHECK:     // Execution Constructs:
!CHECK:     // (12, 3)
!CHECK:     printf(b)
!CHECK:     // (13, 3)
!CHECK:     printf(d)
!CHECK:     // (14, 3)
!CHECK:     a = 20
!CHECK:   }
!CHECK:   // Subroutine
!CHECK:   void sub2() {
!CHECK:     // SubroutineScope, Parent: MainProgramScope
!CHECK:     SymbolTable sub2 {
!CHECK:       // Symbol List:
!CHECK:       // (20, 11)
!CHECK:       // ID: 10, ParentSymbol: 2, ParentSymbolTable: p
!CHECK:       int32 a
!CHECK:       // (18, 14)
!CHECK:       // ID: 9, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, sub2
!CHECK:       int32 b
!CHECK:     }
!CHECK:     // Specification Constructs:
!CHECK:     EntityDeclList {
!CHECK:       {
!CHECK:         // (18, 14)
!CHECK:         NAME: b
!CHECK:         SYMBOL ID: 9
!CHECK:         INIT: 30
!CHECK:       }
!CHECK:     }
!CHECK:     // Execution Constructs:
!CHECK:     // (19, 3)
!CHECK:     printf(b)
!CHECK:     // (20, 3)
!CHECK:     printf(a)
!CHECK:   }
!CHECK: }
