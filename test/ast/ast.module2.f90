
module mod2
  integer:: c(10)
end module mod2

module mod1
  use mod2
  integer :: a = 10 ,b(20)
end module mod1

program pg

  ! Module used
  use mod1

  ! module variables 

  b(1) = 13
  c(9) = 3
  a = a + 10 + b(1)
  
  print *, a, c(9)
end program pg


! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.module2.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (6, 8)
!CHECK:   // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   Module mod1
!CHECK:   // (2, 8)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   Module mod2
!CHECK:   // (11, 9)
!CHECK:   // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() pg
!CHECK: }
!CHECK: Module mod2 { 
!CHECK:   // ModuleScope, Parent: GlobalScope
!CHECK:   SymbolTable mod2 {
!CHECK:     // Symbol List: 
!CHECK:     // (3, 13)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod2
!CHECK:     int32[1:10] c
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:     {
!CHECK:       // (3, 13)
!CHECK:       DIMS: [1:10]
!CHECK:       NAME: c
!CHECK:       SYMBOL ID: 2
!CHECK:     }
!CHECK:   }
!CHECK: }
!CHECK: Module mod1 { 
!CHECK:   // ModuleScope, Parent: GlobalScope
!CHECK:   SymbolTable mod1 {
!CHECK:     // Symbol List: 
!CHECK:     // (8, 14)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod1
!CHECK:     int32 a
!CHECK:     // (8, 22)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod1
!CHECK:     int32[1:20] b
!CHECK:     // (1, 1)
!CHECK:     // ID: 11, ParentSymbol: 10, ParentSymbolTable: mod2
!CHECK:     int32[1:10] c
!CHECK:   }
!CHECK:   UsedSymbolTables {
!CHECK:     // ModuleScope, Parent: None
!CHECK:     SymbolTable mod2 {
!CHECK:       // Symbol List: 
!CHECK:       // (1, 1)
!CHECK:       // ID: 10, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod2
!CHECK:       int32[1:10] c
!CHECK:     }
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   UseStmtList {
!CHECK:     // (7, 7)
!CHECK:     Module mod2
!CHECK:   }
!CHECK:   EntityDeclList {
!CHECK:     {
!CHECK:       // (8, 14)
!CHECK:       NAME: a
!CHECK:       SYMBOL ID: 4
!CHECK:       INIT: 10
!CHECK:     }
!CHECK:     {
!CHECK:       // (8, 22)
!CHECK:       DIMS: [1:20]
!CHECK:       NAME: b
!CHECK:       SYMBOL ID: 5
!CHECK:     }
!CHECK:   }
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 pg() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable pg {
!CHECK:     // Symbol List: 
!CHECK:     // (20, 3)
!CHECK:     // ID: 9, ParentSymbol: 12, ParentSymbolTable: mod1
!CHECK:     int32 a
!CHECK:     // (18, 3)
!CHECK:     // ID: 7, ParentSymbol: 13, ParentSymbolTable: mod1
!CHECK:     int32[1:20] b
!CHECK:     // (19, 3)
!CHECK:     // ID: 8, ParentSymbol: 14, ParentSymbolTable: mod1
!CHECK:     int32[1:10] c
!CHECK:   }
!CHECK:   UsedSymbolTables {
!CHECK:     // ModuleScope, Parent: None
!CHECK:     SymbolTable mod1 {
!CHECK:       // Symbol List: 
!CHECK:       // (1, 1)
!CHECK:       // ID: 12, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod1
!CHECK:       int32 a
!CHECK:       // (1, 1)
!CHECK:       // ID: 13, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod1
!CHECK:       int32[1:20] b
!CHECK:       // (1, 1)
!CHECK:       // ID: 14, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticGlobal, Intent_None, mod2
!CHECK:       int32[1:10] c
!CHECK:     }
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   UseStmtList {
!CHECK:     // (14, 7)
!CHECK:     Module mod1
!CHECK:   }
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (18, 3)
!CHECK:   b(1) = 13
!CHECK:   // (19, 3)
!CHECK:   c(9) = 3
!CHECK:   // (20, 3)
!CHECK:   t.2 = a + 10
!CHECK:   t.1 = t.2 + b(1)
!CHECK:   a = t.1
!CHECK:   // (22, 3)
!CHECK:   printf(a, c(9))
!CHECK: }
