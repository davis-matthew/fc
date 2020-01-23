program v

logical::bool(2)
integer:: a(2,2,2)
integer(kind=8) ::d(2) 

real :: b(2)
double precision ::c(2)

bool(1) = .true.
bool(2) = .false.
a(1, 1,1) = 10
a(2, 1,1) = 20
a(2, 2,1) = 30
a(1, 2,1) = 40
a(1, 1,2) = 50
a(2, 1,2) = 60
a(1, 2,2) = 70
a(2, 2,2) = 80
d(1) = 1000000_8
d(2) = 100_8
b(1) = 3.468947
b(2) = 3000.46
c(1) = 9743589.40_8
c(2) = 0.40_8

write (*, *)bool

write (*, *)a
write (*, *)d

write (*, *)b
write (*, *)c

end program v
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.write_arr_all.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() v
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 v() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable v {
!CHECK:     // Symbol List: 
!CHECK:     // (4, 11)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, v
!CHECK:     int32[1:2, 1:2, 1:2] a
!CHECK:     // (7, 9)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, v
!CHECK:     real[1:2] b
!CHECK:     // (3, 10)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, v
!CHECK:     logical[1:2] bool
!CHECK:     // (8, 20)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, v
!CHECK:     double[1:2] c
!CHECK:     // (5, 19)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, v
!CHECK:     int64[1:2] d
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (10, 1)
!CHECK:   bool(1) = .true.
!CHECK:   // (11, 1)
!CHECK:   bool(2) = .false.
!CHECK:   // (12, 1)
!CHECK:   a(1, 1, 1) = 10
!CHECK:   // (13, 1)
!CHECK:   a(2, 1, 1) = 20
!CHECK:   // (14, 1)
!CHECK:   a(2, 2, 1) = 30
!CHECK:   // (15, 1)
!CHECK:   a(1, 2, 1) = 40
!CHECK:   // (16, 1)
!CHECK:   a(1, 1, 2) = 50
!CHECK:   // (17, 1)
!CHECK:   a(2, 1, 2) = 60
!CHECK:   // (18, 1)
!CHECK:   a(1, 2, 2) = 70
!CHECK:   // (19, 1)
!CHECK:   a(2, 2, 2) = 80
!CHECK:   // (20, 1)
!CHECK:   d(1) = 1000000
!CHECK:   // (21, 1)
!CHECK:   d(2) = 100
!CHECK:   // (22, 1)
!CHECK:   b(1) = 3.468947
!CHECK:   // (23, 1)
!CHECK:   b(2) = 3000.46
!CHECK:   // (24, 1)
!CHECK:   c(1) = 9743589.40
!CHECK:   // (25, 1)
!CHECK:   c(2) = 0.40
!CHECK:    // (27, 1)
!CHECK:   write   bool()
!CHECK:    // (29, 1)
!CHECK:   write   a()
!CHECK:    // (30, 1)
!CHECK:   write   d()
!CHECK:    // (32, 1)
!CHECK:   write   b()
!CHECK:    // (33, 1)
!CHECK:   write   c()
!CHECK: }
