program v

logical::boolT = .TRUE.
logical::boolF = .false.

integer:: a= 10
integer(kind=8) ::d = 9999999999_8

real :: b = 3.45600000
double precision ::c=4.556_8

write (*, *) boolT
write (*, *) boolF

write (*, *) a
write (*, *) d

write (*, *) b
write (*, *) c

end program v
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.write_all.f90
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
!CHECK:     // (6, 11)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, v
!CHECK:     int32 a
!CHECK:     // (9, 9)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, v
!CHECK:     real b
!CHECK:     // (4, 10)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, v
!CHECK:     logical boolf
!CHECK:     // (3, 10)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, v
!CHECK:     logical boolt
!CHECK:     // (10, 20)
!CHECK:     // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, v
!CHECK:     double c
!CHECK:     // (7, 19)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, v
!CHECK:     int64 d
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (3, 10)
!CHECK:   boolt = .TRUE.
!CHECK:   // (4, 10)
!CHECK:   boolf = .false.
!CHECK:   // (6, 11)
!CHECK:   a = 10
!CHECK:   // (7, 19)
!CHECK:   d = 9999999999
!CHECK:   // (9, 9)
!CHECK:   b = 3.45600000
!CHECK:   // (10, 20)
!CHECK:   c = 4.556
!CHECK:    // (12, 1)
!CHECK:   write   boolt
!CHECK:    // (13, 1)
!CHECK:   write   boolf
!CHECK:    // (15, 1)
!CHECK:   write   a
!CHECK:    // (16, 1)
!CHECK:   write   d
!CHECK:    // (18, 1)
!CHECK:   write   b
!CHECK:    // (19, 1)
!CHECK:   write   c
!CHECK: }
