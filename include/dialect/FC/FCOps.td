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
// Defines MLIR fortran operations.
// FIXME:
// 1. Missing verifier , parser functions.
// 2. Summary and description of operations.
// 3. Move non-trivial functions to cpp file.

#ifndef FORTRAN_OPS
#define FORTRAN_OPS

include "mlir/IR/OpBase.td"
include "mlir/Analysis/CallInterfaces.td"
include "mlir/Transforms/LoopLikeInterface.td"

def FC_Dialect : Dialect {
  let name = "fc";
  let cppNamespace = "FC";
}

def FC_RefType : Type<CPred<"$_self.isa<FC::RefType>()">,
    "Ref type">;

def FC_PointerType : Type<CPred<"$_self.isa<FC::PointerType>()">,
    "Pointer type">;

def FC_ArrayType : Type<CPred<"$_self.isa<FC::ArrayType>()">,
    "array type">;

def AnyRef : TypeConstraint<FC_RefType.predicate, "any ref">;

def AnyPointer : TypeConstraint<FC_PointerType.predicate, "any pointer">;

def AnyRefLike : TypeConstraint<Or<[FC_RefType.predicate, FC_PointerType.predicate]>, "any ref like">;

def AnyArray : TypeConstraint<FC_ArrayType.predicate, "any array">;


// Base class for FC dialect ops.
class FC_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<FC_Dialect, mnemonic, traits> {

  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

// Loop related operations.
include "dialect/FC/FCLoopOperations.td"
// High level MLIR operations about Modules, Functions , etc.
include "dialect/FC/FCProgramUnit.td"
// Complex type related operations
include "dialect/FC/ComplexOperations.td"

def FC_AllocaOp : FC_Op<"alloca", [Symbol]> {
  let summary = "memory allocation operation static/dynamic";
  let description = [{
    Allocates a new fc.RefType of specified type with a fixed base pointer location
    in memory.
    Eg.:
    %A = alloc_static() : fc.RefType<i32>
  }];

  let arguments = (ins Variadic<Index>:$bounds, OptionalAttr<AnyAttr>:$value, OptionalAttr<BoolAttr>:$captured);
  let results = (outs FC_RefType);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, StringRef name, FC::RefType ptrType, ArrayRef<Value > boundsArr = {}", [{
       result.types.push_back(ptrType);
       result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
       result.addOperands(boundsArr);
     }]
  >];

  let extraClassDeclaration = [{
    StringRef getName() {
     return getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName())
         .getValue();
    }
    // TODO: Add sideeffects interface.
    bool isCaptured() {
     if (captured().hasValue()) {
       return captured().getValue();
     }
     return false;
    }
    void setCaptured(BoolAttr value) {
      this->setAttr("captured",value);
    }
    RefType getType() { return getResult().getType().cast<RefType>(); }
    operand_range getBounds() { return {operand_begin(), operand_end()}; }
    void setInitializer(Attribute val) {
      this->setAttr("value",val);
    }
  }];

  let verifier = ? ;
  let printer = [{
    p << "fc.allocate ";
    p << getName();
    SmallVector<Value , 2> operands(getBounds());
    if (!operands.empty()) {
      p << '[';
      for (unsigned I = 0; I < operands.size(); ++I) {
        if (I > 0)
          p << ", ";
        p << operands[I];
      }
      p << ']';
    }
    if (auto attr = getAttrOfType<StringAttr>("alloc_kind")) {
      p << ", " << attr.getValue();
    }

    if (auto attr = getAttrOfType<BoolAttr>("captured")) {
      if (attr.getValue()) {
        p << ", implicitly_captured";
      }
    }
    if (value().hasValue()) {
      p  << " {" << value().getValue() << "} ";
    }
    p << " : " << getType();
  }];
  let parser = ? ;
}

def FC_DeallocaOp : FC_Op<"deallocate"> {
  let summary = "memory deallocation operation";
  let description = [{
    The "dealloc" operation frees the region of memory referenced by a ref
    only if it is malloced.
  }];

  let arguments = (ins AnyRef:$ref);

  let verifier = ? ;
  let printer = [{
    p << "fc.deallocate " << ref();
    p.printOptionalAttrDict(getAttrs());
    p << " : " << ref().getType();
  }];
  let parser = ? ;
}


// TODO: Move to ArrayOps.td.
def FC_LBoundOp : FC_Op<"lbound"> {
  let summary = "returns lower bound of current array";
  let description = [{

  }];

  let arguments = (ins AnyRef:$ref, Index: $dim);
  let results = (outs Index);

  let extraClassDeclaration = [{
    int64_t getDim();
    }];

  let verifier = ? ;
  let printer = [{
    p << "fc.lbound " << ref() << ", " << dim();
    p.printOptionalAttrDict(getAttrs());
    p << " : " << ref().getType();
  }];
  let parser = ? ;
}

def FC_UBoundOp : FC_Op<"ubound"> {
  let summary = "returns lower bound of current array";
  let description = [{

  }];

  let arguments = (ins AnyRef:$ref, Index: $dim);
  let results = (outs Index);

  let extraClassDeclaration = [{
    int64_t getDim();
    }];

  let verifier = ? ;
  let printer = [{
    p << "fc.ubound " << ref() << ", " << dim();
    p.printOptionalAttrDict(getAttrs());
    p << " : " << ref().getType();
  }];
  let parser = ? ;
}

// NOTE: This is not an elementwise comapre. This is like strcmp.
// FIXME: Convert to array operations. return boolean array.
def FC_ArrayCmpIOp : FC_Op<"array_cmpi"> {
  let summary = "Compares two integer like arrays and returns boolean result";
  let description = [{

  }];

  let arguments = (ins AnyArray:$lhs, AnyArray:$rhs);
  let results = (outs I1);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, CmpIPredicate predicate,"
    "Value lhs, Value rhs", [{
      result.addOperands({lhs, rhs});
      result.types.push_back(builder->getIntegerType(1));
      result.addAttribute(
      CmpIOp::getPredicateAttrName(),
      builder->getI64IntegerAttr(static_cast<int64_t>(predicate)));
    }]>];

  let extraClassDeclaration = [{
    CmpIPredicate getPredicate() {
      return (CmpIPredicate)getAttrOfType<IntegerAttr>(CmpIOp::getPredicateAttrName())
          .getInt();
    }

  }];

  // TODO: add compatibility tests.
  let verifier = ? ;
  let printer = [{
    p << "fc.array_cmpi ";
    p << stringifyCmpIPredicate(getPredicate());
    p << " " << lhs() << ", " << rhs();
    p << " : " << getResult().getType();
  }];
  let parser = ? ;
}

// NOTE: This is not an elementwise comapre. This is like strcmp.
// FIXME: Convert to array operations. return boolean array.
def FC_ArrayCmpFOp : FC_Op<"array_cmpf"> {
  let summary = "Compares two integer like arrays and returns boolean result";
  let description = [{

  }];

  let arguments = (ins AnyArray:$lhs, AnyArray:$rhs);
  let results = (outs I1);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, CmpFPredicate predicate,"
    "Value lhs, Value rhs", [{
      result.addOperands({lhs, rhs});
      result.types.push_back(builder->getIntegerType(1));
      result.addAttribute(
      CmpFOp::getPredicateAttrName(),
      builder->getI64IntegerAttr(static_cast<int64_t>(predicate)));
    }]>];

  let extraClassDeclaration = [{
    CmpFPredicate getPredicate() {
      return (CmpFPredicate)getAttrOfType<IntegerAttr>(CmpFOp::getPredicateAttrName())
          .getInt();
    }

  }];

  // TODO: add compatibility tests.
  let verifier = ? ;
  let printer = [{
    p << "fc.array_cmpf ";
    p << getCmpFPredicateNames()[(unsigned)getPredicate()];
    p << " " << lhs() << ", " << rhs();
    p << " : " << getResult().getType();
  }];
  let parser = ? ;
}

// TODO: Move to ArrayOps.td
class FC_ArrayBinOp<string mnemonic, list<OpTrait> traits = []> :
  FC_Op<mnemonic, traits> {

  let summary = "Element wise binary operation on arrays, returns array of same type.";
  let description = [{

  }];

  let arguments = (ins AnyArray:$opd1, AnyArray:$opd2);
  let results = (outs AnyArray);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value opd1, Value opd2",
    [{
      result.addOperands({opd1, opd2});
      result.types.push_back(opd2.getType());
    }]>];

  // TODO: add compatibility tests.
  let verifier = ? ;
  let printer = ? ;
  let parser = ? ;
}

def FC_ArrayAddIOp : FC_ArrayBinOp<"array_addi", [Commutative]>;
def FC_ArrayAddFOp : FC_ArrayBinOp<"array_addf", []>;
def FC_ArraySubIOp : FC_ArrayBinOp<"array_subi", []>;
def FC_ArraySubFOp : FC_ArrayBinOp<"array_subf", []>;
def FC_ArrayMulIOp : FC_ArrayBinOp<"array_muli", [Commutative]>;
def FC_ArrayMulFOp : FC_ArrayBinOp<"array_mulf", []>;
def FC_ArrayDivIOp : FC_ArrayBinOp<"array_divi", []>;
def FC_ArrayDivFOp : FC_ArrayBinOp<"array_divf", []>;
def FC_ArrayModIOp : FC_ArrayBinOp<"array_modi", []>;
def FC_ArrayModFOp : FC_ArrayBinOp<"array_modf", []>;

def FC_FCLoadOp : FC_Op<"load", []> {
  let summary = "load operation";
  let description = [{FC load pointer type operation}];

  let results = (outs AnyType);

  let arguments = (ins AnyRefLike : $pointer, Variadic<Index>:$indices);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value pointerVal, ArrayRef<SubscriptRange> indices = {}">];

  let extraClassDeclaration = [{
    mlir::Value getPointer() { return pointer(); }
    operand_range getIndices() { return {operand_begin() + 1, operand_end()}; }
    Type getType() { return getResult().getType(); }
    void getSubscriptRangeList(SubscriptRangeList &subsList) {}
    }];

  let verifier = ? ;
  let printer = [{
    p << "fc.load " << getPointer();
    SmallVector<Value , 2> operands(getIndices());
    if (!operands.empty()) {
      p << '[';
      for (unsigned I = 0; I < operands.size(); ++I) {
        if (I > 0) {
          p << ", ";
        }
        p << operands[I];
      }
      p << ']';
    }
    p.printOptionalAttrDict(getAttrs());
    p << " : " << getType();
  }];
  let parser = ? ;
}

def FC_FCStoreOp : FC_Op<"store", []> {
  let summary = "store operation";
  let description = [{FC store for pointer type}];
  let arguments = (ins AnyType:$valueToStore, AnyRefLike : $pointer, Variadic<Index>:$indices);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value valueToStore, Value pointerVal, ArrayRef<SubscriptRange> indices = {}">];

  let extraClassDeclaration = [{
    mlir::Value getPointer() { return pointer(); }
    mlir::Value getValueToStore() { return valueToStore(); }
    operand_range getIndices() { return {operand_begin() + 2, operand_end()}; }
    }];

  let verifier = ? ;
  let printer = [{
    p << "fc.store " << getValueToStore();
    p << ", " << getPointer();
    SmallVector<Value , 2> operands(getIndices());
    if (!operands.empty()) {
      p << '[';
      for (unsigned I = 0; I < operands.size(); ++I) {
        if (I > 0) {
          p << ", ";
        }
        p << operands[I];
      }
      p << ']';
    }
    p.printOptionalAttrDict(getAttrs());
    p << " : " << getPointer().getType();
  }];
  let parser = ? ;
}

// TODO: Dead code. Remove it.
def FC_ExtractValueOp : FC_Op<"extractvalue", []> {
  let summary = "extractvalue operation";
  let description = [{FC extractvalue arrval type operation}];

  let results = (outs AnyType);

  let arguments = (ins AnyArray : $arrval, Variadic<Index>:$indices);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value arrval, ArrayRef<SubscriptRange> indices = {}">];

  let extraClassDeclaration = [{
    mlir::Value getArrayVal() { return arrval(); }
    operand_range getIndices() { return {operand_begin() + 1, operand_end()}; }
    Type getType() { return getResult().getType(); }
    void getSubscriptRangeList(SubscriptRangeList &subsList) {}
    }];

  let verifier = ? ;
  let printer = [{
    p << "fc.extractvalue " << getArrayVal() << '[';
    SmallVector<Value , 2> operands(getIndices());
    if (!operands.empty()) {
      for (unsigned I = 0; I < operands.size() - 1; ++I) {
        p << operands[I] << ", ";
      }
      p << operands.back();
    }
    p << ']';
    p.printOptionalAttrDict(getAttrs());
    p << " : " << getType();
  }];
  let parser = ? ;
}

// TODO: Dead code. Remove it.
def FC_InsertValueOp : FC_Op<"insertvalue", []> {
  let summary = "insertvalue operation";
  let description = [{FC insertvalue for arrval type}];
  let arguments = (ins AnyType:$valueToStore, AnyArray : $arrval, Variadic<Index>:$indices);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value valueToStore, Value arrval, ArrayRef<SubscriptRange> indices = {}">];

  let extraClassDeclaration = [{
    mlir::Value getArrayVal() { return arrval(); }
    mlir::Value getValueToStore() { return valueToStore(); }
    operand_range getIndices() { return {operand_begin() + 2, operand_end()}; }
    }];

  let verifier = ? ;
  let printer = [{
    p << "fc.insertvalue " << getValueToStore();
    p << ", " << getArrayVal() << '[';
    SmallVector<Value , 2> operands(getIndices());
    if (!operands.empty()) {
      for (unsigned I = 0; I < operands.size() - 1; ++I) {
        p << operands[I] << ", ";
      }
      p << operands.back();
    }
    p << ']';
    p.printOptionalAttrDict(getAttrs());
    p << " : " << getArrayVal().getType();
  }];
  let parser = ? ;
}

def FC_PrintOp : FC_Op<"print", []> {
  let summary = "print operation";
  let description = [{Operation for Fortran print statement }];
  let arguments = (ins Variadic<AnyType> : $args);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"Builder *builder, OperationState &result, "
                            "ArrayRef<Value > args">];
  let extraClassDeclaration = [{
    operand_range getArgs() { return args(); }
    }];

  let verifier = ? ;
  let printer = [{
    p << "fc.print ";
    SmallVector<Value , 2> operands(getArgs());
    if (!operands.empty()) {
      for (unsigned I = 0; I < operands.size() - 1; ++I) {
        p << operands[I] << ", ";
      }
      p << operands.back();
    }
    p.printOptionalAttrDict(getAttrs());
  }];
  let parser = ? ;
}

def FC_WriteOp : FC_Op<"write", []> {
  let summary = "write operation";
  let description = [{FC write operation}];

  let arguments = (ins I32:$unit, Variadic<AnyType>:$args);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value unit,"
    "ArrayRef<Value > args = {}", [{
      result.addOperands(unit);
      result.addOperands(args);
    }]>];

  let verifier = ? ;
  let printer = [{ p << getOperationName() << " ";
    SmallVector<Value , 2> operands(operand_begin(), operand_end());
    if (!operands.empty()) {
      for (unsigned I = 0; I < operands.size() - 1; ++I) {
        p << operands[I] << ", ";
      }
      p << operands.back();
    }
    p.printOptionalAttrDict(getAttrs());
  }];
  let parser = ? ;
}

def FC_ReadOp : FC_Op<"read", []> {
  let summary = "read operation";
  let description = [{FC read operation}];

  let arguments = (ins I32:$unit, Variadic<AnyType>:$args);
  let results = (outs I32);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value unit,"
    "ArrayRef<Value > args = {}", [{
      result.addOperands(unit);
      result.addOperands(args);
    }]>];

  let verifier = ? ;
  let printer = [{ p << getOperationName() << " ";
    SmallVector<Value , 2> operands(operand_begin(), operand_end());
    if (!operands.empty()) {
      for (unsigned I = 0; I < operands.size() - 1; ++I) {
        p << operands[I] << ", ";
      }
      p << operands.back();
    }
    p.printOptionalAttrDict(getAttrs());
  }];
  let parser = ? ;
}

// Way to represent the constant string.
// Returns character array.
def FC_StringOp : FC_Op<"constant_string", []> {
  let summary = "string operation";
  let description = [{ FC constant string}];

  let arguments = (ins StrAttr:$value);
  // FIXME: More specific type.
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, StringAttr attr", [{
      result.addAttribute("value", attr);
    }]>];

  let verifier = ? ;
  let printer = [{ p << getOperationName()
                      << "(\"" << value() << "\")";
                    p << ": " << getResult().getType();
                }];

  let parser = ? ;
}

def FC_OpenOp : FC_Op<"open", []> {
  let summary = "open operation";
  let description = [{FC open operation}];

  let arguments = (ins I32:$unit, AnyType:$file);
  let results = (outs I32);


  let verifier = ? ;
  let printer = [{ p << getOperationName() << " ";
    SmallVector<Value , 2> operands(operand_begin(), operand_end());
    if (!operands.empty()) {
      for (unsigned I = 0; I < operands.size() - 1; ++I) {
        p << operands[I] << ", ";
      }
      p << operands.back();
    }
    p.printOptionalAttrDict(getAttrs());
  }];
  let parser = ? ;
}

def FC_CloseOp : FC_Op<"close", []> {
  let summary = "close operation";
  let description = [{FC close operation}];

  let arguments = (ins I32:$unit);
  let results = (outs I32);


  let verifier = ? ;
  let printer = [{ p << getOperationName() << " "  << unit();
    p.printOptionalAttrDict(getAttrs());
  }];
  let parser = ? ;
}

// Used for fc::ast::CastExpr
def FC_CastOp : FC_Op<"cast", []> {
  let summary = "Cast operation";
  let description = [{FC cast expression}];
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value v", [{
      result.addOperands(v);
    }]>];

  let verifier = ?;
  let printer = ?;
  let parser = ?;
}

// MemRefCastOp is a nicer name, but it's already used by standard-ops.
def FC_CastToMemRefOp : FC_Op<"cast_to_memref", []> {
  let summary = "Experimental MemRef cast operation";
  let description = [{Converts the fc.Reftype to std.memref}];
  let arguments = (ins AnyRef:$fcRef);
  let results = (outs AnyMemRef);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value ref">];

  let extraClassDeclaration = [{
    mlir::Value getRef() { return fcRef(); }
  }];

  let verifier = ?;
  let printer = [{
    p << "fc.cast_to_memref " << getRef() << " : "
      << getResult().getType();
  }];

  let parser = ?;
}


// Similar to LLVMAddressOfOp.
// TODO: rename as global_address_of_op.
def FC_AddressOfOp
    : FC_Op<"addressOf", []> {

  let summary = " Address of Operation";
  let description = [{Global Address of operation}];
  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs AnyRef);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, RefType resType,"
              "StringRef name, ArrayRef<NamedAttribute> attrs = {}", [{
      result.addAttribute("global_name", builder->getSymbolRefAttr(name));
      result.addAttributes(attrs);
      result.addTypes(resType);}]>,

    OpBuilder<"Builder *builder, OperationState &result, GlobalOp global, "
              "ArrayRef<NamedAttribute> attrs = {}", [{
      build(builder, result, global.getType(), global.sym_name(),
            attrs);}]>
  ];

  let extraClassDeclaration = [{
    RefType getType() {
      return getResult().getType().cast<RefType>();
    }
    /// Return the fc.global operation that defined the value referenced
    /// here.
    GlobalOp getGlobal();
  }];

  let verifier = ?;
  let printer = [{
    p << "fc.addressOf @";
    p << global_name();
    p << " : " << getType();
  }];
  let parser = ?;
}

def FC_GlobalOp : FC_Op<"global", [Symbol]> {
  let summary = "Global Operation";
  let description = [{Global operation}];
  let arguments = (ins TypeAttr: $type, StrAttr: $sym_name, OptionalAttr<AnyAttr>:$value);
  let results = (outs AnyRef);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, RefType type, "
              "bool isConstant, StringRef name, "
              "Attribute value">
  ];
  let regions = (region AnyRegion:$initializer);

  let extraClassDeclaration = [{
    /// Return the MLIR type of the global.
    FC::RefType getType() {
      return type().cast<FC::RefType>();
    }
    /// Return the initializer attribute if it exists, or a null attribute.
    Attribute getValueOrNull() {
      return value().getValueOr(Attribute());
    }

    /// Return the initializer region. This may be empty, but if it is not it
    /// terminates in an `std.return` op with the initializer value.
    Region &getInitializerRegion() {
      return getOperation()->getRegion(0);
    }
    /// Return the initializer block. If the initializer region is empty this
    /// is nullptr. If it is not nullptr, it terminates with an `std.return`
    /// op with the initializer value.
    Block *getInitializerBlock() {
      return getInitializerRegion().empty() ?
        nullptr : &getInitializerRegion().front();
    }

  }];

  let verifier = ?;
  let printer = [{
    p << "fc.global " << sym_name() << " { ";
    auto attr = getValueOrNull();
    if (attr)
      p.printAttribute(attr);
    else
      p << "undef";
    p << " } : " << getType();
    if (getInitializerBlock())
      p.printRegion(getInitializerRegion());
  }];
  let parser = ?;
}

// TODO: Fix name:. This is like llvm::GEP for
// IO related operations.
def FC_ArrayEleOp : FC_Op<"arrayele", []> {
  let summary = "ArrayEle Operation";
  let description = [{FC Array Element operation}];
  let arguments = (ins AnyRef:$pointer, Variadic<Index>:$indices);
  let results = (outs AnyRef);

  let extraClassDeclaration = [{
    mlir::Value getPointer() { return pointer(); }
    }];

  let verifier = ?;
  let printer = ?;
  let parser = ?;
}

def FC_UndefOp : FC_Op<"undef", []> {
  let summary = "Undef Operation";
  let description = [{Undef value operation of any type}];
  let results = (outs AnyType);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, RefType resType", [{
      result.addTypes(resType);}]>
  ];

  let verifier = ?;
  let printer = ?;
  let parser = ?;
}

def FC_StoIAOp : FC_Op<"stoia", []> {
  let summary = "FCStoIAOp Operation";
  let description = [{FC string to int array op}];
  let arguments = (ins AnyRef:$unit, AnyRef:$expr);

  let verifier = ?;
  let printer = ?;
  let parser = ?;
}

def FC_StoIOp : FC_Op<"stoi", []> {
  let summary = "FCStoIOp Operation";
  let description = [{FC char to int op}];
  let arguments = (ins AnyRef:$unit);
  let results = (outs I32);

  let verifier = ?;
  let printer = ?;
  let parser = ?;
}

def FC_StrCpyOp : FC_Op<"strcpy", []> {
  let summary = "FC_StringOp Operation";
  let description = [{FC string copy operation}];
  let arguments = (ins AnyRef:$dest, AnyRef:$src);

  let verifier = ?;
  let printer = ?;
  let parser = ?;
}

def FC_ItoSOp : FC_Op<"itos", []> {
  let summary = "FC_ItoSOp  Operation";
  let description = [{FC int to string operation}];
  let arguments = (ins AnyRef:$unit, I32:$expr);

  let verifier = ?;
  let printer = [{
    p << "fc.stoi " << unit() << ", ";
    p << expr();
  }];
  let parser = ?;
}

def FC_SprintfOp : FC_Op<"sprintf", []> {
  let summary = "sprintf operation";
  let description = [{FC sprintf operation}];

  let arguments = (ins AnyRef:$unit, Variadic<AnyType> : $args);

  let verifier = ? ;
  let printer = [{
    p << "fc.sprintf "<< unit() << ", ";
    SmallVector<Value , 2> operands(args());
    if (!operands.empty()) {
      for (unsigned I = 0; I < operands.size() - 1; ++I) {
        p << operands[I] << ", ";
      }
      p << operands.back();
    }
    p.printOptionalAttrDict(getAttrs());
  }];
  let parser = ? ;
}

def FC_StrCatOp : FC_Op<"strcat", []> {
  let summary = "FC_StrCatOp Operation";
  let description = [{FC string concat operation}];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType);

  let verifier = ?;
  let printer = [{ p << getOperationName() << " ";
                   p << lhs() << ", ";
                   p << rhs() << " () -> ";
                   p << getResult().getType();
                   }];

  let parser = ?;
}

def FC_TrimOp : FC_Op<"trim", []> {
  let summary = "FC_TrimOp Operation";
  let description = [{FC trim operation}];
  let arguments = (ins AnyType:$str);

  let verifier = ?;
  let printer = [{ p << getOperationName() << " ";
                   p << str() << " () -> ";
                   p << "()";
                   }];

  let parser = ?;
}

// TODO: Can this be expressed as normal variables?
def FC_ArgcOp : FC_Op<"argc", []> {
  let summary = "FC_Argc Operation";
  let description = [{FC Argc operation}];
  let results = (outs I32);

  let verifier = ?;
  let printer = [{ p << getOperationName();
                   p << "() ";
                   p << " -> ";
                   p << getResult().getType();
                   }];

  let parser = ?;
}

// TODO: Can this be expressed as normal variables?
def FC_ArgvOp : FC_Op<"argv", []> {
  let summary = "FC_Argv Operation";
  let description = [{FC Argv operation}];
  let arguments = (ins I32:$pos, AnyType:$str);

  let verifier = ?;
  let printer = [{ p << getOperationName();
                   p << "[" << pos() << "]";
                   p << ", ";
                   p << str() << " () -> ";
                   p << "()";
                   }];

  let parser = ?;
}

// Get the pointer for the given ref type.
def FC_GetPointerToOp : FC_Op<"get_pointer_to", []> {
  let summary = "Get Pointer to Operation";
  let description = [{Used for fortran pointer assignments}];
  let arguments = (ins AnyRef:$fcRef);
  let results = (outs AnyPointer);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value ref">];

  let extraClassDeclaration = [{
    mlir::Value getRef() { return fcRef(); }
  }];

  let verifier = ?;
  let printer = [{
    p << "fc.get_pointer_to " << getRef() << " : "
      << getResult().getType();
  }];

  let parser = ?;
}

def FC_NullPointerOp : FC_Op<"constant_null_pointer"> {
  let summary = "Generates null pointer";
  let description = [{

  }];

  let results = (outs AnyPointer);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, PointerType resType", [{
      result.addTypes(resType);}]>
  ];
  let verifier = ? ;
  let printer = [{
    p << "fc.constant_null_pointer : " << getResult().getType();
  }];
  let parser = ? ;
}

// Get the pointer for the given ref type.
def FC_CmpPointerEqualOp : FC_Op<"cmp_pointer_equal", []> {
  let summary = "Compares two pointer";
  let description = [{Used for fortran pointer assignments}];
  let arguments = (ins AnyPointer:$lhs, AnyPointer:$rhs);
  let results = (outs I1);

  let verifier = ?;
  let printer = [{
    p << "fc.cmp_pointer_equal " << lhs() << ", " << rhs() << " : "
      << getResult().getType();
  }];

  let parser = ?;
}

#endif // FORTRAN_OPS
