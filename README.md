# **FC - MLIR Based fortran frontend**

A new front end for Fortran has been written in the design spirit of LLVM/Clang. Approximately 40000+ lines of code effort has got close to Fortran 95 standard. The Front end is complete end to end solution without semantic analysis. MLIR is designed as a high level abstraction to support source aware compiler  optimizations. It was primarily designed for Tensorflow. In this work we have utilized the MLIR framework to support a Fortran compiler. We have designed a new dialect in MLIR for high level optimizations in Fortran language. Using few constructs we have demonstrated the capability to support OpenMP in the framework. Apart from 400+ unit test cases, 2 spec 2017 benchmarks are passing to validate the framework.

### Dependencies
```
1. Clang 7.0.1 -http://releases.llvm.org/download.html
2. llvm-project - https://github.com/compiler-tree-technologies/llvm-project.
3. gfortran  7.1.0 (for running unit tests)
```

## Build instructions
```

1. LLVM/MLIR : There are some MLIR changes done for FC
  git clone https://github.com/compiler-tree-technologies/llvm-project -b fc

2. Building LLVM:
Enable MLIR project while building LLVM. For llvm build instructions, refer to
llvm-project README

3. Download clang.7.0.1 compiler for building FC.

4. FC clone
  git clone https://github.com/compiler-tree-technologies/fc

5. Set PATH.

export PATH=${CLANG_701}/bin:$PATH

6. mkdir <fc-build-path>/build && cd build

7. FC build instructions (shared libs build doesn't work, LLD-7.0.1 is much faster compared to /bin/ld)
cmake -G Ninja -DLLVM_DIR=${LLVM_PROJECT_BUILD}/lib/cmake/llvm/ ../fc \
 -DCMAKE_INSTALL_PREFIX=<install-prefix> -DCMAKE_C_COMPILER=clang \
 -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_FLAGS="-fuse-ld=lld" \
 -DCMAKE_CXX_FLAGS="-fuse-ld=lld"

8. ninja install

9. Set following environment variables to run

export PATH=${FC_BUILD}/bin:$PATH
export LD_LIBRARY_PATH=${FC_BUILD}/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=${FC_BUILD}build/lib:$LIBRARY_PATH
export CLANG_BINARY=${CLANG_701}/bin/clang

10. ninja test to run eixsting tests.

```

### Notes
1. Shared libs build doesn't work (MLIR libs has some issues)
2. ninja test to run the test suite.
3. During FC build cmake will look for mlir-tblgen in ${DLLVM_DIR}/../../../bin . If not found build will fail.

## Running HelloWorld

```
$ cat hello.f90
program hello
  print *, "Hello world!"
end program hello
$ <path/to/fc>/bin/fc hello.f90
$ ./a.out
Hello world!
```

## Intermediate Representations

```
1. Emit AST
 $ fc <inputfile> -emit-ast
2. Emit high level MLIR IR
 $ fc <inputfile> -emit-ir
3. Emit low level MLIR IR
 $ fc <inputfile> -emit-mlir
4. Emit LLVM IR
 $ fc <inputfile> -emit-llvm
5. Emit ASM
 $ fc <inputfile> -emit-asm
```

## Fortran standard
Fortran  standard can be specified via -std=f77 or -std=f95. If not specified standard is derived from file extension. Source files with ".f" are compiled to F77 standard and all other extensions are compiled to default standard.

### Help
To display other commandline options use below command.
```
fc --help
```

## Developed by
[CompilerTree Technologies](http://compilertree.com)
