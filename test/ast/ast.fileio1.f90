program readtest
  integer, dimension(10) :: x
  integer, dimension(10) :: y
  integer :: i
  

   open  (1, file = './fileio/input/1.dat', status = 'old')
   do i = 1,10 
      read(1,*) x(i), y(i)
   end do 

   do i = 1,10  
      print *, x(i), y(i)
   end do 

end program readtest
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.fileio1.f90
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
!CHECK:     // (4, 14)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, readtest
!CHECK:     int32 i
!CHECK:     // (2, 29)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, readtest
!CHECK:     int32[1:10] x
!CHECK:     // (3, 29)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, readtest
!CHECK:     int32[1:10] y
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (7, 4)
!CHECK:   1 = open({./fileio/input/1.dat}, OLD)
!CHECK:   // (8, 4)
!CHECK:   t.1 = (/*IndVar=*/i, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:   do (t.1) {
!CHECK:     // (9, 7)
!CHECK:     read(unit = 1) x(i), y(i)
!CHECK:   }
!CHECK:   // (12, 4)
!CHECK:   t.2 = (/*IndVar=*/i, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:   do (t.2) {
!CHECK:     // (13, 7)
!CHECK:     printf(x(i), y(i))
!CHECK:   }
!CHECK: }
