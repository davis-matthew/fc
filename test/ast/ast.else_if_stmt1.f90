program t
  integer::a = 10, b = 20
 
  if (a /= 10) then
   a = 30
  else if (a == 10) then
   a = 20
  else
    a = 40
  end if
  print *,a
end program t
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.else_if_stmt1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() t
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 t() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable t {
!CHECK:     // Symbol List: 
!CHECK:     // (2, 12)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, t
!CHECK:     int32 a
!CHECK:     // (2, 20)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, t
!CHECK:     int32 b
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (2, 12)
!CHECK:   a = 10
!CHECK:   // (2, 20)
!CHECK:   b = 20
!CHECK:   // (4, 3)
!CHECK:   t.1 = a != 10
!CHECK:   if (t.1) {
!CHECK:     // (5, 4)
!CHECK:     a = 30
!CHECK:   }
!CHECK:   // (6, 8)
!CHECK:   t.2 = a == 10
!CHECK:   else if (t.2) {
!CHECK:     // (7, 4)
!CHECK:     a = 20
!CHECK:   }
!CHECK:   // (9, 5)
!CHECK:   else {
!CHECK:     // (9, 5)
!CHECK:     a = 40
!CHECK:   }
!CHECK:   // (11, 3)
!CHECK:   printf(a)
!CHECK: }
