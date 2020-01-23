program pgm
complex :: a
a = (1,2)
print *, a
end
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: complex_expr1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() pgm
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 pgm() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable pgm {
!CHECK:     // Symbol List:
!CHECK:     // (2, 12)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, pgm
!CHECK:     complex(8) a
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (3, 1)
!CHECK:   a = (1, 2)
!CHECK:   // (4, 1)
!CHECK:   printf(a)
!CHECK: }
