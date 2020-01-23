module mod
  integer, parameter :: rank=3, rank2=rank*rank, total=rank2*(rank2+1)/2
  logical, parameter :: log = rank > total
end module mod

program foo
  use mod
  integer :: k
  integer :: arr1(rank2) = (/ (k, k = 1, rank2) /)

  do k = 1, (total - rank - 35)
    arr1(k) = 12
  end do

  if ((rank + total) < (rank2)) then
    arr1(rank) = 2
  end if

  if (.true. .and. (1 > rank)) then
    if (.false. .eqv. log) then
      arr1(rank + k) = 2
    end if
  end if
end program foo
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: constprop1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (6, 9)
!CHECK:   // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() foo
!CHECK:   // (1, 8)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   Module mod
!CHECK: }
!CHECK: Module mod {
!CHECK:   // ModuleScope, Parent: GlobalScope
!CHECK:   SymbolTable mod {
!CHECK:     // Symbol List:
!CHECK:     // (3, 25)
!CHECK:     // ID: 5, Constant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod
!CHECK:     logical log
!CHECK:     // (2, 25)
!CHECK:     // ID: 2, Constant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod
!CHECK:     int32 rank
!CHECK:     // (2, 33)
!CHECK:     // ID: 3, Constant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod
!CHECK:     int32 rank2
!CHECK:     // (2, 50)
!CHECK:     // ID: 4, Constant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod
!CHECK:     int32 total
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:     {
!CHECK:       // (2, 25)
!CHECK:       NAME: rank
!CHECK:       SYMBOL ID: 2
!CHECK:       INIT: 3
!CHECK:     }
!CHECK:     {
!CHECK:       // (2, 33)
!CHECK:       NAME: rank2
!CHECK:       SYMBOL ID: 3
!CHECK:       INIT: 9
!CHECK:     }
!CHECK:     {
!CHECK:       // (2, 50)
!CHECK:       NAME: total
!CHECK:       SYMBOL ID: 4
!CHECK:       INIT: 45
!CHECK:     }
!CHECK:     {
!CHECK:       // (3, 25)
!CHECK:       NAME: log
!CHECK:       SYMBOL ID: 5
!CHECK:       INIT: .false.
!CHECK:     }
!CHECK:   }
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 foo() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable foo {
!CHECK:     // Symbol List:
!CHECK:     // (9, 14)
!CHECK:     // ID: 9, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32[1:9] arr1
!CHECK:     // (8, 14)
!CHECK:     // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     int32 k
!CHECK:     // (20, 23)
!CHECK:     // ID: 12, ParentSymbol: 16, ParentSymbolTable: mod
!CHECK:     logical log
!CHECK:     // (11, 22)
!CHECK:     // ID: 11, ParentSymbol: 13, ParentSymbolTable: mod
!CHECK:     int32 rank
!CHECK:     // (9, 19)
!CHECK:     // ID: 8, ParentSymbol: 14, ParentSymbolTable: mod
!CHECK:     int32 rank2
!CHECK:     // (11, 14)
!CHECK:     // ID: 10, ParentSymbol: 15, ParentSymbolTable: mod
!CHECK:     int32 total
!CHECK:   }
!CHECK:   UsedSymbolTables {
!CHECK:     // ModuleScope, Parent: None
!CHECK:     SymbolTable mod {
!CHECK:       // Symbol List:
!CHECK:       // (1, 1)
!CHECK:       // ID: 16, Constant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod
!CHECK:       logical log
!CHECK:       // (1, 1)
!CHECK:       // ID: 13, Constant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod
!CHECK:       int32 rank
!CHECK:       // (1, 1)
!CHECK:       // ID: 14, Constant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod
!CHECK:       int32 rank2
!CHECK:       // (1, 1)
!CHECK:       // ID: 15, Constant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod
!CHECK:       int32 total
!CHECK:     }
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   UseStmtList {
!CHECK:     // (7, 7)
!CHECK:     Module mod
!CHECK:   }
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (9, 14)
!CHECK:   t.1 = (/*IndVar=*/k, /*Init=*/1, /*End=*/9, /*Incr=*/1)
!CHECK:   arr1 = (/ t.1 /)
!CHECK:   // (11, 3)
!CHECK:   t.2 = (/*IndVar=*/k, /*Init=*/1, /*End=*/7, /*Incr=*/1)
!CHECK:   do (t.2) {
!CHECK:     // (12, 5)
!CHECK:     arr1(k) = 12
!CHECK:   }
!CHECK:   // (15, 3)
!CHECK:   if (.false.) {
!CHECK:     // (16, 5)
!CHECK:     arr1(3) = 2
!CHECK:   }
!CHECK:   // (19, 3)
!CHECK:   if (.false.) {
!CHECK:     // (20, 5)
!CHECK:     if (.true.) {
!CHECK:       // (21, 7)
!CHECK:       t.3 = 3 + k
!CHECK:       arr1(t.3) = 2
!CHECK:     }
!CHECK:   }
!CHECK: }
