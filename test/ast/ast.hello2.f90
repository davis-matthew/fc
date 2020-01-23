program test
      character(len=80) :: string
      character(len=20) :: string2
      string = "hello world"
      string2 = string
      print *, string2, string, string2
end program test
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.hello2.f90
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
!CHECK:     character[0:20] string2
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (4, 7)
!CHECK:   string() = {hello world}
!CHECK:   // (5, 7)
!CHECK:   string2() = string()
!CHECK:   // (6, 7)
!CHECK:   printf(string2(), string(), string2())
!CHECK: }
