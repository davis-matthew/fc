program readtest
  character(len=80) ::msg
  
  open  (1, file = './fileio/input/3.dat', status = 'old')
  read (1, *) msg
  print *, msg
  close(1)

end program readtest
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: fileio3.f90
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
!CHECK:     // (2, 23)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, readtest
!CHECK:     character[0:80] msg
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (4, 3)
!CHECK:   1 = open({./fileio/input/3.dat}, OLD)
!CHECK:   // (5, 3)
!CHECK:   read(unit = 1) msg()
!CHECK:   // (6, 3)
!CHECK:   printf(msg())
!CHECK:   // (7, 3)
!CHECK:   1 = close(/*unit=*/1)
!CHECK: }
