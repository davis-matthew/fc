program test
      integer :: a = (2*3) * 2 * 2
      integer :: b = 2 * 3 * 2
      
      if ( a / b == 2) THEN
        a = 11
      END IF
      
      PRINT *,a
end program test
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.if_stmt3.f90
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
!CHECK:     // (2, 18)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 a
!CHECK:     // (3, 18)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 b
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (2, 18)
!CHECK:   a = 24
!CHECK:   // (3, 18)
!CHECK:   b = 12
!CHECK:   // (5, 7)
!CHECK:   t.2 = a / b
!CHECK:   t.1 = t.2 == 2
!CHECK:   if (t.1) {
!CHECK:     // (6, 9)
!CHECK:     a = 11
!CHECK:   }
!CHECK:   // (9, 7)
!CHECK:   printf(a)
!CHECK: }
