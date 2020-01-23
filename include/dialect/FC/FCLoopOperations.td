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
// Represents the Fortran do loop. can be viewed as higher level loops

// compared to affine.for / loop.for.
// TODO:
// 1. Use AffineMap for bounds.
// 2. Return udpated induction variable as SSA value
// 3. Allow multiple input/output SSA values.
// 4. Allow cycle/exit/return operations inside.

def FC_DoOp : FC_Op<"do",
      [DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       SingleBlockImplicitTerminator<"mlir::loop::TerminatorOp">]> {
  let summary = "do operation";
  let description = [{
  }];
  let arguments = (ins Variadic<AnyType>:$args,  StrAttr:$construct_name);  // Should be isIntOrIndex()
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, StringAttr constructNameAttr, "
              "Value lowerBound, Value upperBound, Value step", [{
    result.addOperands({lowerBound, upperBound, step});
    Region *bodyRegion = result.addRegion();
    DoOp::ensureTerminator(*bodyRegion, *builder, result.location);
    bodyRegion->front().addArgument(builder->getIndexType());
    result.addAttribute("construct_name",constructNameAttr);
   }]>,

   OpBuilder<"Builder *builder, OperationState &result, StringAttr constructNameAttr", [{
     Region *bodyRegion = result.addRegion();
     DoOp::ensureTerminator(*bodyRegion, *builder, result.location);
     bodyRegion->front().addArgument(builder->getIndexType());
     result.addAttribute("construct_name",constructNameAttr);
    }]>
  ];

  let extraClassDeclaration = [{
    Block *getBody() { return &region().front(); }
    Value getIndVar() { return getBody()->getArgument(0); }
    StringRef getConstructName() { return construct_name(); }
    mlir::OpBuilder getBodyBuilder() {
      return mlir::OpBuilder(getBody(), std::prev(getBody()->end()));
    }
    void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
    void setStep(Value step) {
      getOperation()->setOperand(2, step); }
    mlir::Value getLowerBound() {
      if (args().empty()) return nullptr;
      return getOperation()->getOperand(0); }
    mlir::Value getUpperBound() {
      if (args().empty()) return nullptr;
      return getOperation()->getOperand(1); }
    mlir::Value getStep() {
      if (args().empty()) return nullptr;
      return getOperation()->getOperand(2); }
  }];

  let verifier = ?;
  let printer = [{
    SmallVector<Value , 2> operands(args());
    p << "fc.do ";
    if (!operands.empty()) {
      p << getIndVar() << " = " << operands[0] << ", ";
      p << operands[1] << ", " << operands[2];
    }
    p << " {construct_name = \"" << construct_name() << "\"} ";
    p.printRegion(region(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
    p << " enddo";
    p.printOptionalAttrDict(getAttrs());
  }];
  let parser = ?;
}

// TODO: Should this also be a terminator?
// FIXME: How to pass the SSA value operands as output.
def FC_ExitOp : FC_Op<"exit", []> {
  let summary = "exit operation";
  let description = [{Fortran exit operation}];

  let arguments = (ins StrAttr:$construct_name);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"Builder *builder, OperationState &result, StringAttr constructNameAttr", [{
     result.addAttribute("construct_name",constructNameAttr);
  }]>,
  ];

  let extraClassDeclaration = [{
    StringRef getConstructName() { return construct_name(); }
  }];

  let verifier = ? ;
  let printer = [{
    p << "fc.exit " << construct_name();
  }];
  let parser = ? ;
}

// TODO: Should this also be a terminator?
// FIXME: How to pass the SSA value operands as output.
def FC_CycleOp : FC_Op<"cycle", []> {
  let summary = "cycle operation";
  let description = [{Fortran cycle operation}];

  let arguments = (ins StrAttr:$construct_name);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"Builder *builder, OperationState &result, StringAttr constructNameAttr", [{
     result.addAttribute("construct_name",constructNameAttr);
  }]>,
  ];

  let extraClassDeclaration = [{
    StringRef getConstructName() { return construct_name(); }
  }];

  let verifier = ? ;
  let printer = [{
    p << "fc.cycle " << construct_name();
  }];
  let parser = ? ;
}

// TODO: hacky operation for return from do loop.
// FIXME: Use fc.return operation instead.
def FC_DoReturnOp : FC_Op<"do_return", []> {
  let summary = "return operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<"Builder *b, OperationState &result", [{ }]>,
    OpBuilder<"Builder *b, OperationState &result, mlir::Value valueToReturn", [{
      result.addOperands(valueToReturn);
    }]>
    ];

  let verifier = ?;
  let printer = [{
    p << getOperationName() << " ";
    if (!operands().empty())
      p << *operand_begin();
  }];

  let parser = ?;
}

// Value(s) to pass to the next iteration of a DO loop.
// FIXME: Not used yet.
def FC_DoNextOp : FC_Op<"do_next", [Terminator]> {
  let summary = "do_next operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<"Builder *b, OperationState &result", [{ }]>,
    // TODO: accept list or Values
    OpBuilder<"Builder *b, OperationState &result, mlir::Value nextValue", [{
      result.addOperands(nextValue);
    }]>
    ];

  let verifier = ?;
  let printer = [{
    p << getOperationName() << " ";
    if (!operands().empty())
      p << *operand_begin();
  }];

  let parser = ?;
}
