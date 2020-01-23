

module mod1
  integer :: a = 10
end module mod1

program pg

  ! Module used
  use mod1

  ! module variables 
  print *, a
end program pg


! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.basic_module.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (3, 8)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   Module mod1
!CHECK:   // (7, 9)
!CHECK:   // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() pg
!CHECK: }
!CHECK: Module mod1 { 
!CHECK:   // ModuleScope, Parent: GlobalScope
!CHECK:   SymbolTable mod1 {
!CHECK:     // Symbol List: 
!CHECK:     // (4, 14)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod1
!CHECK:     int32 a
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:     {
!CHECK:       // (4, 14)
!CHECK:       NAME: a
!CHECK:       SYMBOL ID: 2
!CHECK:       INIT: 10
!CHECK:     }
!CHECK:   }
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 pg() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable pg {
!CHECK:     // Symbol List: 
!CHECK:     // (13, 12)
!CHECK:     // ID: 4, ParentSymbol: 5, ParentSymbolTable: mod1
!CHECK:     int32 a
!CHECK:   }
!CHECK:   UsedSymbolTables {
!CHECK:     // ModuleScope, Parent: None
!CHECK:     SymbolTable mod1 {
!CHECK:       // Symbol List: 
!CHECK:       // (1, 1)
!CHECK:       // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod1
!CHECK:       int32 a
!CHECK:     }
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   UseStmtList {
!CHECK:     // (10, 7)
!CHECK:     Module mod1
!CHECK:   }
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (13, 3)
!CHECK:   printf(a)
!CHECK: }
