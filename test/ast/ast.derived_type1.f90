program foo
  type :: PowerTyp                   ! Data type for POWER
    integer           :: stride              ! Output stride
    integer           :: skip                ! Output skip
    character(len=80) :: filenamebase        ! Base name for output file
    integer        :: nofreq                 ! Number of frequencies to compute
    integer        :: window                 ! Start timestep for window function
    integer        :: y_index                ! Y-index
    type(PowerCoords), pointer :: coords     ! Pointer to the coordinates
    integer        :: nocoords               ! Number of coords
    type(PowerTyp), pointer    :: next       ! Pointer to the next entry
    logical        :: mine                   !Do I own any twinkles in this plane
  end type PowerTyp

  type :: PowerCoords                ! The coordinates for POWER
    integer, dimension(7)      :: coords     ! [normal i_min i_max j_min ...]
    type(PowerCoords), pointer :: next       ! Pointer to the next coordinates
  end type PowerCoords

  type(PowerTyp) :: pt
  type(PowerCoords) :: pc
end program foo
! RUN: %fc %s -emit-ast -o - | FileCheck %s
!CHECK: Program: ast.derived_type1.f90
!CHECK: // GlobalScope, Parent: None
!CHECK: SymbolTable Global {
!CHECK:   // Symbol List: 
!CHECK:   // (1, 9)
!CHECK:   // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
!CHECK:   (int32)() foo
!CHECK: }
!CHECK: // MainProgram
!CHECK: int32 foo() {
!CHECK:   // MainProgramScope, Parent: GlobalScope
!CHECK:   SymbolTable foo {
!CHECK:     // Symbol List: 
!CHECK:     // (21, 24)
!CHECK:     // ID: 15, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     dt.foo.PowerCoords pc
!CHECK:     // (20, 21)
!CHECK:     // ID: 14, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, foo
!CHECK:     dt.foo.PowerTyp pt
!CHECK:   }
!CHECK:   // Specification Constructs: 
!CHECK:   DerivedTypeDef powertyp {
!CHECK:     // MainProgramScope, Parent: MainProgramScope
!CHECK:     SymbolTable PowerTyp {
!CHECK:       // Symbol List: 
!CHECK:       // (9, 35)
!CHECK:       // ID: 8, NonConstant, NonAllocatable, NonTarget, Pointer, StaticLocal, Intent_None, PowerTyp
!CHECK:       dt.foo.PowerCoords* coords
!CHECK:       // (5, 26)
!CHECK:       // ID: 4, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, PowerTyp
!CHECK:       character[0:80] filenamebase
!CHECK:       // (12, 23)
!CHECK:       // ID: 11, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, PowerTyp
!CHECK:       logical mine
!CHECK:       // (11, 35)
!CHECK:       // ID: 10, NonConstant, NonAllocatable, NonTarget, Pointer, StaticLocal, Intent_None, PowerTyp
!CHECK:       dt.foo.PowerTyp* next
!CHECK:       // (10, 23)
!CHECK:       // ID: 9, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, PowerTyp
!CHECK:       int32 nocoords
!CHECK:       // (6, 23)
!CHECK:       // ID: 5, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, PowerTyp
!CHECK:       int32 nofreq
!CHECK:       // (4, 26)
!CHECK:       // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, PowerTyp
!CHECK:       int32 skip
!CHECK:       // (3, 26)
!CHECK:       // ID: 2, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, PowerTyp
!CHECK:       int32 stride
!CHECK:       // (7, 23)
!CHECK:       // ID: 6, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, PowerTyp
!CHECK:       int32 window
!CHECK:       // (8, 23)
!CHECK:       // ID: 7, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, PowerTyp
!CHECK:       int32 y_index
!CHECK:     }
!CHECK:   }
!CHECK:   DerivedTypeDef powercoords {
!CHECK:     // MainProgramScope, Parent: MainProgramScope
!CHECK:     SymbolTable PowerCoords {
!CHECK:       // Symbol List: 
!CHECK:       // (16, 35)
!CHECK:       // ID: 12, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, PowerCoords
!CHECK:       int32[1:7] coords
!CHECK:       // (17, 35)
!CHECK:       // ID: 13, NonConstant, NonAllocatable, NonTarget, Pointer, StaticLocal, Intent_None, PowerCoords
!CHECK:       dt.foo.PowerCoords* next
!CHECK:     }
!CHECK:   }
!CHECK:   EntityDeclList {
!CHECK:   }
!CHECK: }
