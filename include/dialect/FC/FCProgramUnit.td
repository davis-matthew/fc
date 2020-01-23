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


// Operation to represent the Fortran Modules. Semantics are similar to function
// but it is not callable and contains only one block.
def FC_FortranModuleOp : FC_Op<"fortran_module", [NativeOpTrait<"IsIsolatedFromAbove">,
                         Symbol, SymbolTable,
                         SingleBlockImplicitTerminator<"FC::FCReturnOp">]> {
  let summary = "OPeration for representing fortran module";

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, StringRef name", [{
    Region *bodyRegion = result.addRegion();
    result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
    FortranModuleOp::ensureTerminator(*bodyRegion, *builder, result.location);
   }]>
  ];

  let extraClassDeclaration = [{
    static FortranModuleOp create(Location location, StringRef name) {
       OperationState state(location, "fc.fortran_module");
       Builder builder(location->getContext());
       FC::FortranModuleOp::build(&builder, state, name);
       return llvm::cast<FortranModuleOp>(Operation::create(state));
    }
    void push_back(Operation* op) {
      auto terminator = body().front().getTerminator();
      body().front().push_back(op);
      op->moveBefore(terminator);
    }
    StringRef getName() {
     return getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName())
         .getValue();
    }

  }];

  let verifier = ?;
  let printer = [{
    p << "fc.fortran_module " << getName();
    p.printRegion(body(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  }];
  let parser = ?;
}

// Function operation to represent Fortran Subroutine, functions, MainProgram
// TODO: Should contain information such as "use"d modules, captured variables from
// different scope, etc.
def FC_FCFuncOp
    : FC_Op<"function",
                        [NativeOpTrait<"IsIsolatedFromAbove">,
                         Symbol, SymbolTable, NativeOpTrait<"FunctionLike">]> {
  let summary = "FC dialect function";

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, StringRef name, "
              "FunctionType type, ArrayRef<NamedAttribute> attrs = {}">
  ];

  let extraClassDeclaration = [{
    Block *addEntryBlock();
    Block *addBlock() {
      assert(!empty() && "function should at least have an entry block");
      push_back(new Block());
      return &back();
    }
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }
    unsigned getNumFuncResults() { return getType().getResults().size(); }
    FunctionType getType() {
      return getAttrOfType<TypeAttr>(getTypeAttrName())
          .getValue()
          .cast<FunctionType>();
    }
    void addNestedFunction(FC::FCFuncOp funcOp);
    static FC::FCFuncOp create(Location location, StringRef name, FunctionType type,
                      ArrayRef<NamedAttribute> attrs ={});
  }];

  let printer = [{
    FunctionType fnType = getType();
    p << "fc.function @" << getName();
    impl::printFunctionSignature(p, *this, fnType.getInputs(),
                                 /*isVariadic=*/false, fnType.getResults());
    if (!isExternal())
      p.printRegion(body(),false);
  }];
  let verifier = [{
    if (isExternal()) {
      return success();
    }
    // Verify that the argument list of the function and the arg list of the entry
    // block line up.  The trait already verified that the number of arguments is
    // the same between the signature and the block.
    auto fnInputTypes = getType().getInputs();
    Block &entryBlock = body().front();
    for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
      if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
        return emitOpError("type of entry block argument #")
               << i << '(' << entryBlock.getArgument(i).getType()
               << ") must match the type of the corresponding argument in "
               << "function signature(" << fnInputTypes[i] << ')';

    return success();
  }];
  let parser = ?;
}

// Operation to represent Subroutine and Fortran function calls.
def FC_FCCallOp : FC_Op<"call", [CallOpInterface]> {
  let summary = "FC call operation";
  let description = [{
    The "call" operation represents a direct call to a function.

  }];

  let arguments = (ins Variadic<AnyType>:$operands, SymbolRefAttr:$symbol_scope, I32Attr:$num_operands);
  let results = (outs Variadic<AnyType>);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, FCFuncOp callee, ArrayRef<Value > operands = {}">,
    OpBuilder<
    "Builder *builder, OperationState &result, SymbolRefAttr symbolScope,"
    "ArrayRef<Type> results, ArrayRef<Value > operands = {}">
  ];

  let extraClassDeclaration = [{
    SymbolRefAttr getCallee() { return symbol_scope(); }
    FunctionType getCalleeType() {
      SmallVector<Type, 4> resultTypes(getResultTypes());
      SmallVector<Type, 8> argTypes(getOperandTypes());
      return FunctionType::get(argTypes, resultTypes, getContext());
    }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let verifier = ? ;
  let printer = [{
    p << "fc.call ";
    p.printAttribute(symbol_scope());
    p <<  "(";
    SmallVector<Value , 2> ops(getArgOperands());
    for (unsigned I = 0; I < ops.size(); ++I) {
      if (I > 0)
       p << ", ";
      p << ops[I];
    }
    p << ")";
    //p.printOptionalAttrDict(getAttrs());
    for (auto type : getResultTypes())
      p << " : " << type  << ", ";
  }];
  let parser = ? ;
}

// Operation to represent the variable access from different
// SymbolTable / Scope. Currently, the scope is represented using the
// mlir::SymbolTable and mlir::SymRefAttr.
// Example: Fortran module variable access inside an external subroutine.
// Idea is to retain the high level information of the program.
// This information can be well utilized by Alias Analysis like framework.
def FC_GetElementRefOp : FC_Op<"get_element_ref", []> {
  let summary = "FC Get element ref operation";
  let description = [{
    Get the element from the different scope.

  }];

  let arguments = (ins SymbolRefAttr:$symbol_scope);
  let results = (outs AnyRef);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, SymbolRefAttr symbolScope,"
    "Type resultType">
  ];

  let extraClassDeclaration = [{
    SymbolRefAttr getSymRef() { return symbol_scope(); }
  }];

  let verifier = ? ;
  let printer = [{
    p << "fc.get_element_ref ";
    p.printAttribute(symbol_scope());
    p << " : " << getResult().getType();
  }];
  let parser = ? ;
}

// Terminator Op for fc.function.
// NOTE: This can be used inside the nested regions as well.
// TODO: Follow up with the latest discussion in mlir-dev.
def FC_FCReturnOp : FC_Op<"return", [Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<
    "Builder *b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];

  let verifier = ?;
  let printer = [{
    p << "fc.return ";
    for (auto operand : operands()) {
      p << operand << " ";
    }
  }];
  let parser = ?;
}

// Operation to indicate that the argument passed as operand
// is captured by the nested subroutine.
// "BlockArgument" is not a "Symbol".
// This operations provides a way access function argument as Symbol
// TODO: Can we do something better here?
def FC_CaptureArgOp : FC_Op<"capture_func_argument", [Symbol]> {
  let summary = "return operation";
  let description = [{
    Dummy operation to create a symbol for function arguments.
  }];

  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *b, OperationState &result, BlockArgument arg,StringRef name", [{
       result.addOperands(arg);
       result.addTypes(arg.getType());
       result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      b->getStringAttr(name));
      }]
  >];

  let extraClassDeclaration = [{
    StringRef getName() {
     return getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName())
         .getValue();
    }
  }];

  let verifier = ?;
  let printer = [{
    p << "fc.capture_func_argument " << operand() << ", name: " << getName();
  }];
  let parser = ?;
}