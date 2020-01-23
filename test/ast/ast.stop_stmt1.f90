program i
  integer, parameter::a = 4
  integer, parameter::b = 10

  stop (a+b)
  
  
end program i
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.stop_stmt1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() i
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 i() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable i {
!CHECK:     // Symbol List: 
!CHECK:     // (2, 23)
!CHECK:     // ID: 2, Constant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     int32 a
!CHECK:     // (3, 23)
!CHECK:     // ID: 3, Constant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     int32 b
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (2, 23)
!CHECK:   a = 4
!CHECK:   // (3, 23)
!CHECK:   b = 10
!CHECK:   // (5, 3)
!CHECK:   stop 14
!CHECK: }
