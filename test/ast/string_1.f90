program test
  integer, parameter :: b = 80
  character(len=b) :: msg1 = "hello world"
  character(len=80) :: msg2
  character :: ch1 = "c"
  character :: ch2
  integer :: a
  msg2 = "Hello world"
  ch2 = "C"
  a = 6 + 4
end program test
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: string_1.f90
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
!CHECK:     // (7, 14)
!CHECK:     // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 a
!CHECK:     // (2, 25)
!CHECK:     // ID: 2, Constant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     int32 b
!CHECK:     // (5, 16)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     character ch1
!CHECK:     // (6, 16)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     character ch2
!CHECK:     // (3, 23)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     character[0:80] msg1
!CHECK:     // (4, 24)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     character[0:80] msg2
!CHECK:   }
!CHECK:   // Specification Constructs:
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs:
!CHECK:   // (2, 25)
!CHECK:   b = 80
!CHECK:   // (3, 23)
!CHECK:   msg1 = {hello world}
!CHECK:   // (5, 16)
!CHECK:   ch1 = c
!CHECK:   // (8, 3)
!CHECK:   msg2() = {Hello world}
!CHECK:   // (9, 3)
!CHECK:   ch2 = C
!CHECK:   // (10, 3)
!CHECK:   a = 10
!CHECK: }
