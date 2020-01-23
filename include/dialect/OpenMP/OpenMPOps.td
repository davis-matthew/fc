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
include "mlir/Transforms/LoopLikeInterface.td"

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
  let regions = (region AnyRegion:$region);

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

def OMP_SingleOp : OMP_Op < "single",
       []> {
  let summary = "openmp operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$symbols);
  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;
  let builders = [
   OpBuilder<"Builder *builder, OperationState &result", [{
     result.addRegion();
    }]>
  ];

  let extraClassDeclaration = [{
    Block *getBody() { return &region().front(); }
    Region *getRegion() { return &region(); }
    }];

  let verifier = ?;
  let printer = [{
    p << "omp.single(";
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

def OMP_MasterOp : OMP_Op < "master",
       []> {
  let summary = "!$omp master construct";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$symbols);
  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;
  let builders = [
   OpBuilder<"Builder *builder, OperationState &result", [{
     result.addRegion();
    }]>
  ];

  let extraClassDeclaration = [{
    Block *getBody() { return &region().front(); }
    Region *getRegion() { return &region(); }
    }];

  let verifier = ?;
  let printer = [{
    p << "omp.master(";
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

def OMP_OmpDoOp : OMP_Op<"do", []> {
  let summary = "omp do operation";
  let description = [{
  }];
  let arguments = (ins Variadic<AnyType>:$args);
  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, ValueRange args", [{
    result.addOperands(args);
    Region *bodyRegion = result.addRegion();
    if (bodyRegion->empty()) {
      Block *b = new Block();
      bodyRegion->getBlocks().insert(bodyRegion->begin(), b);
    }
   }]>];

  let extraClassDeclaration = [{
    constexpr static unsigned int IVINDEX = 0;
    constexpr static unsigned int LBINDEX = 1;
    constexpr static unsigned int UBINDEX = 2;
    constexpr static unsigned int STEPINDEX = 3;
    Block *getBody() { return &region().front(); }
    Value getIndVar() { return getOperation()->getOperand(IVINDEX); }
    mlir::OpBuilder getBodyBuilder() {
      return mlir::OpBuilder(getBody(), std::prev(getBody()->end()));
    }
    void setLowerBound(Value bound) {
      getOperation()->setOperand(LBINDEX, bound);
    }
    void setUpperBound(Value bound) {
      getOperation()->setOperand(UBINDEX, bound);
    }
    void setStep(Value step) {
      getOperation()->setOperand(STEPINDEX, step);
    }
    mlir::Value getLowerBound() {
      if (args().empty()) return nullptr;
      return getOperation()->getOperand(LBINDEX);
    }
    mlir::Value getUpperBound() {
      if (args().empty()) return nullptr;
      return getOperation()->getOperand(UBINDEX);
    }
    mlir::Value getStep() {
      if (args().empty()) return nullptr;
      return getOperation()->getOperand(STEPINDEX);
    }
    mlir::Region *getRegion() { return &region(); }
  }];


  let verifier = ?;
  let printer = [{
    SmallVector<Value , 2> operands(args());
    p << "omp.do";
    if (!operands.empty()) {
      p << "(";
      p << getIndVar() << " = " << getLowerBound() << ", ";
      p << getUpperBound() << ", " << getStep();
      p << ")";
    }
    p << "(";
    for (unsigned i = 4; i < operands.size(); ++i) {
      p << operands[i];
      if (i != operands.size() - 1)
        p << ", ";
    }

    p << ")";
    p.printRegion(region(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
    p << " enddo";
    p.printOptionalAttrDict(getAttrs());
  }];
  let parser = ?;
}

// Is not a loop like interface because
// 1. MLIR has passes optimises loop like interfaces.
//    Should OpenMP loops be optimized currently ?
// 2. If we add IV as block argument, it doesn't get
//    converted from mlir::Type to LLVM::LLVMType.
// TODO : This should be treated as [IsolatedFromAbove] ?
//        Since we are not using block arguments this can not
//        be used currently
def OMP_ParallelDoOp : OMP_Op<"parallel_do", []> {
  let summary = "omp parallel do operation";
  let description = [{
  }];
  let arguments = (ins Variadic<AnyType>:$args);
  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, ValueRange args", [{
    result.addOperands(args);
    Region *bodyRegion = result.addRegion();
    if (bodyRegion->empty()) {
      Block *b = new Block();
      bodyRegion->getBlocks().insert(bodyRegion->begin(), b);
    }
   }]>];

  let extraClassDeclaration = [{
    unsigned ivIndex = 0;
    unsigned lbIndex = 1;
    unsigned ubIndex = 2;
    unsigned stepIndex = 3;
    Block *getBody() { return &region().front(); }
    Value getIndVar() { return getOperation()->getOperand(ivIndex); }
    mlir::OpBuilder getBodyBuilder() {
      return mlir::OpBuilder(getBody(), std::prev(getBody()->end()));
    }
    void setLowerBound(Value bound) {
      getOperation()->setOperand(lbIndex, bound);
    }
    void setUpperBound(Value bound) {
      getOperation()->setOperand(ubIndex, bound);
    }
    void setStep(Value step) {
      getOperation()->setOperand(stepIndex, step);
    }
    mlir::Value getLowerBound() {
      if (args().empty()) return nullptr;
      return getOperation()->getOperand(lbIndex);
    }
    mlir::Value getUpperBound() {
      if (args().empty()) return nullptr;
      return getOperation()->getOperand(ubIndex);
    }
    mlir::Value getStep() {
      if (args().empty()) return nullptr;
      return getOperation()->getOperand(stepIndex);
    }
    mlir::Region *getRegion() { return &region(); }
  }];

  let verifier = ?;
  let printer = [{
    SmallVector<Value , 2> operands(args());
    p << "omp.parallel_do";
    if (!operands.empty()) {
      p << "(";
      p << getIndVar() << " = " << getLowerBound() << ", ";
      p << getUpperBound() << ", " << getStep();
      p << ")";
    }
    p << "(";
    for (unsigned i = 4; i < operands.size(); ++i) {
      p << operands[i];
      if (i != operands.size() - 1)
        p << ", ";
    }

    p << ")";
    p.printRegion(region(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
    p << " enddo";
    p.printOptionalAttrDict(getAttrs());
  }];
  let parser = ?;
}
#endif // FORTRAN_OPS
