program readtest
      character(len=34) :: msg
      read *, msg
      print *, msg
end program readtest
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.string.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() readtest
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 readtest() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable readtest {
!CHECK:     // Symbol List: 
!CHECK:     // (2, 28)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, readtest
!CHECK:     character[0:34] msg
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (3, 7)
!CHECK:   read msg()
!CHECK:   // (4, 7)
!CHECK:   printf(msg())
!CHECK: }
