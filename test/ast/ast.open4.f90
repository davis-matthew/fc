program outputdata   

   real, dimension(10) :: x
   real, dimension(10) :: y
   integer :: i

   open(1, file = 'data1.dat', status='old')  
   do i = 1,10  
      read(1,*) x(i), y(i)
      print *, x(i), y(i)
   end do  
   close(1)

   
end program outputdata
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.open4.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() outputdata
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 outputdata() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable outputdata {
!CHECK:     // Symbol List: 
!CHECK:     // (5, 15)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, outputdata
!CHECK:     int32 i
!CHECK:     // (3, 27)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, outputdata
!CHECK:     real[1:10] x
!CHECK:     // (4, 27)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, outputdata
!CHECK:     real[1:10] y
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (7, 4)
!CHECK:   1 = open({data1.dat}, OLD)
!CHECK:   // (8, 4)
!CHECK:   t.1 = (/*IndVar=*/i, /*Init=*/1, /*End=*/10, /*Incr=*/1)
!CHECK:   do (t.1) {
!CHECK:     // (9, 7)
!CHECK:     read(unit = 1) x(i), y(i)
!CHECK:     // (10, 7)
!CHECK:     printf(x(i), y(i))
!CHECK:   }
!CHECK:   // (12, 4)
!CHECK:   1 = close(/*unit=*/1)
!CHECK: }
