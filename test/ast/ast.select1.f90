program selectCaseProg
implicit none

   !local variable declaration
   integer, parameter :: a = 1008

   select case (a)
   case (1)
     print *, "Number is 1"
   case (2)
     print *, "Number is 2"
   case (3)
     print *, "Number is 3"
   case (4)
     print *, "Number is 4"
   case (5)
     print *, "Number is 5"
   case default
     print *, "Some other number", a
   end select


end program selectCaseProg

! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.select1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() selectcaseprog
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 selectcaseprog() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable selectCaseProg {
!CHECK:     // Symbol List: 
!CHECK:     // (5, 26)
!CHECK:     // ID: 2, Constant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, selectCaseProg
!CHECK:     int32 a
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (5, 26)
!CHECK:   a = 1008
!CHECK:   // (8, 4)
!CHECK:   if (.false.) {
!CHECK:     // (9, 6)
!CHECK:     printf({Number is 1})
!CHECK:   }
!CHECK:   // (10, 4)
!CHECK:   else if (.false.) {
!CHECK:     // (11, 6)
!CHECK:     printf({Number is 2})
!CHECK:   }
!CHECK:   // (12, 4)
!CHECK:   else if (.false.) {
!CHECK:     // (13, 6)
!CHECK:     printf({Number is 3})
!CHECK:   }
!CHECK:   // (14, 4)
!CHECK:   else if (.false.) {
!CHECK:     // (15, 6)
!CHECK:     printf({Number is 4})
!CHECK:   }
!CHECK:   // (16, 4)
!CHECK:   else if (.false.) {
!CHECK:     // (17, 6)
!CHECK:     printf({Number is 5})
!CHECK:   }
!CHECK:   // (18, 4)
!CHECK:   else {
!CHECK:     // (19, 6)
!CHECK:     printf({Some other number}, 1008)
!CHECK:   }
!CHECK: }
