program arr2
  integer:: a(10,10, 40), b(25, 23, 21, -4:23)
  integer:: i,j,k,l
  i = 1
  j = 2
  k = 3
  l = -2
  b(i,j,k,l) = 30
  b(5,j,k,l) = 10
  a(i+j, k + l, b(i,j,k,l) + i) = b(5,j,k,l) * j
  print *,a(3, 1, 31)
end program arr2
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.arr_assign3.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() arr2
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 arr2() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable arr2 {
!CHECK:     // Symbol List: 
!CHECK:     // (2, 13)
!CHECK:     // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, arr2
!CHECK:     int32[1:10, 1:10, 1:40] a
!CHECK:     // (2, 27)
!CHECK:     // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, arr2
!CHECK:     int32[1:25, 1:23, 1:21, -4:23] b
!CHECK:     // (3, 13)
!CHECK:     // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, arr2
!CHECK:     int32 i
!CHECK:     // (3, 15)
!CHECK:     // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, arr2
!CHECK:     int32 j
!CHECK:     // (3, 17)
!CHECK:     // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, arr2
!CHECK:     int32 k
!CHECK:     // (3, 19)
!CHECK:     // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, arr2
!CHECK:     int32 l
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK:   // Execution Constructs: 
!CHECK:   // (4, 3)
!CHECK:   i = 1
!CHECK:   // (5, 3)
!CHECK:   j = 2
!CHECK:   // (6, 3)
!CHECK:   k = 3
!CHECK:   // (7, 3)
!CHECK:   l = -2
!CHECK:   // (8, 3)
!CHECK:   b(i, j, k, l) = 30
!CHECK:   // (9, 3)
!CHECK:   b(5, j, k, l) = 10
!CHECK:   // (10, 3)
!CHECK:   t.1 = b(5, j, k, l) * j
!CHECK:   t.2 = i + j
!CHECK:   t.3 = k + l
!CHECK:   t.4 = b(i, j, k, l) + i
!CHECK:   a(t.2, t.3, t.4) = t.1
!CHECK:   // (11, 3)
!CHECK:   printf(a(3, 1, 31))
!CHECK: }
