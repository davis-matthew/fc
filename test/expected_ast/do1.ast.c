Program: do1.f90
// GlobalScope, Parent: None
SymbolTable Global {
  // Symbol List: 

  // (1, 9)
  // ID: 1, NonConstant, NonAllocatable, NonTarget, NonPointer, Alloc_None, Intent_None, Global
  (int32)() i
}

// MainProgram
int32 i() {
  // MainProgramScope, Parent: GlobalScope
  SymbolTable i {
    // Symbol List: 

    // (5, 14)
    // ID: 3, NonConstant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
    int32 j
    // (4, 25)
    // ID: 2, Constant, NonAllocatable, NonTarget, NonPointer, StaticLocal, Intent_None, i
    int32 n
  }

  // Specification Constructs: 

  EntityDeclList {
  }

  // Execution Constructs: 

  // (4, 25)
  n = 10
  // (7, 3)
  omp parallel do {
    // (7, 3)
    t.1 = (/*IndVar=*/j, /*Init=*/1, /*End=*/10, /*Incr=*/1)
    do (t.1) {
      // (9, 6)
      printf(j)
    }
  }
}

