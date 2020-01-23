program readtest
  integer, dimension(10) :: x
  
  open  (1, file = './fileio/input/2.dat', status = 'old')
  read (1, *) x
  print *, x
  close(1)

end program readtest
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: fileio2.f90
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
!CHECK:     // (2, 29)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, readtest
!CHECK:     int32[1:10] x
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (4, 3)
!CHECK:   1 = open({./fileio/input/2.dat}, OLD)
!CHECK:   // (5, 3)
!CHECK:   read(unit = 1) x()
!CHECK:   // (6, 3)
!CHECK:   printf(x())
!CHECK:   // (7, 3)
!CHECK:   1 = close(/*unit=*/1)
!CHECK: }
