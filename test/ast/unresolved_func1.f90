
program readtest
integer :: a(20)
  real ::b
b = count1(a)

print *,a,b
end program readtest
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: unresolved_func1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List:
!CHECK:   // (2, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() readtest
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 readtest() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable readtest {
!CHECK:     // Symbol List:
!CHECK:     // (3, 12)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, readtest
!CHECK:     int32[1:20] a
!CHECK:     // (4, 10)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, readtest
!CHECK:     real b
!CHECK:     // (5, 5)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, readtest
!CHECK:     (real)(...) count1
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (5, 1)
!CHECK:   b = count1(a())
!CHECK:   // (7, 1)
!CHECK:   printf(a(), b)
!CHECK: }
