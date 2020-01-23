program readtest
  character ::msg(7)
  integer i
  
  open  (1, file = './fileio/input/4.dat', status = 'old')
  read (1, *) msg
  print *, msg
  close(1)

end program readtest
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.fileio4.f90
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
!CHECK:     // (3, 11)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, readtest
!CHECK:     int32 i
!CHECK:     // (2, 15)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, readtest
!CHECK:     character[1:7] msg
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (5, 3)
!CHECK:   1 = open({./fileio/input/4.dat}, OLD)
!CHECK:   // (6, 3)
!CHECK:   read(unit = 1) msg()
!CHECK:   // (7, 3)
!CHECK:   printf(msg())
!CHECK:   // (8, 3)
!CHECK:   1 = close(/*unit=*/1)
!CHECK: }
