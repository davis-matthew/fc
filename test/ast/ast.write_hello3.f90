program test
      character(len=80) :: string
      character(len=80) :: string2
      string = "in this test case we are trying to print really long message!!!!"
      string2 = string
      write(*, *) string2
      write(*, *) string
end program test
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.write_hello3.f90
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
!CHECK:     // (2, 28)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     character[0:80] string
!CHECK:     // (3, 28)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, test
!CHECK:     character[0:80] string2
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (4, 7)
!CHECK:   string() = {in this test case we are trying to print really long message!!!!}
!CHECK:   // (5, 7)
!CHECK:   string2() = string()
!CHECK:    // (6, 7)
!CHECK:   write   string2()
!CHECK:    // (7, 7)
!CHECK:   write   string()
!CHECK: }
