program i
   integer, parameter::a= ((3 + 4) * 2) 
   integer, parameter::b= 3 + 6 / 3 * 3 + 2 
   stop a + b
end program i
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.stop_stmt3.f90
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
!CHECK:     // (2, 24)
!CHECK:     // ID: 2, Constant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     int32 a
!CHECK:     // (3, 24)
!CHECK:     // ID: 3, Constant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
!CHECK:     int32 b
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (2, 24)
!CHECK:   a = 14
!CHECK:   // (3, 24)
!CHECK:   b = 11
!CHECK:   // (4, 4)
!CHECK:   stop 25
!CHECK: }
