// Copyright (c) 2019, Compiler Tree Technologies Pvt Ltd.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#ifndef OPNEMP_OPS
#define OPENMP_OPS

include "mlir/IR/OpBase.td"

def OMP_Dialect : Dialect {
  let name = "omp";
  let cppNamespace = "OMP";
}

// Base class for OMP dialect ops.
class OMP_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<OMP_Dialect, mnemonic, traits> {

  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def OMP_ParallelOp : OMP_Op < "parallel",
       []> {
  let summary = "openmp operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$symbols);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
   OpBuilder<"Builder *builder, OperationState &result, ArrayRef<Value> args", [{
     result.addRegion();
     result.addOperands(args);
     //ParallelOp::ensureTerminator(*bodyRegion, *builder, result.location);
    }]>
  ];

  let extraClassDeclaration = [{
    Block *getBody() { return &region().front(); }
    Region *getRegion() { return &region(); }
    }];

  let verifier = ?;
  let printer = [{
    p << "omp.parallel(";
    SmallVector<Value, 2> operands(operand_begin(), operand_end());
    if (!operands.empty()) {
      for (unsigned I = 0; I < operands.size() - 1; ++I) {
        p << operands[I] << ", ";
      }
      p << operands.back();
    }
    p << ") ";
    p.printRegion(region(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
    }];

  let parser = ?;
}

def OpenMPTerminatorOp :
   OMP_Op<"terminator", [Terminator]> {
  let summary = "OpenMP terminator operation";
  let description = [{
  }];

  // No custom parsing/printing form.
  let parser = ?;
  let printer = ?;

  // Fully specified by traits.
  let verifier = ?;
}

#endif // FORTRAN_OPS
