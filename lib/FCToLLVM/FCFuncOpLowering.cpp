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
#include "FCToLLVM/FCToLLVMLowering.h"
#include "dialect/FCOps/FCOps.h"

#include "llvm/ADT/Sequence.h"

#define PASS_NAME "FCOPsLowering"
#define DEBUG_TYPE PASS_NAME

using namespace fcmlir;
using namespace mlir;
using namespace FC;

// NOTE: fc.function to LLVM lowering. Not used in the regular flow.
// ProgramUnitLoweringPass should have been called previously. But this works
// too.

struct FCFuncOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;
  FCTypeConverter *converter;

public:
  explicit FCFuncOpLowering(MLIRContext *_context, FCTypeConverter *converter)
      : ConversionPattern(FC::FCFuncOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        converter(converter) {}

  // NOTE: The whole function is copied from mlir::FuncOp To LLVM Converter.

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = cast<FC::FCFuncOp>(op);
    FunctionType type = funcOp.getType();
    // Pack the result types into a struct.
    Type packedResult;
    if (type.getNumResults() != 0)
      if (!(packedResult = converter->packFunctionResults(type.getResults())))
        return matchFailure();
    LLVM::LLVMType resultType = packedResult
                                    ? packedResult.cast<LLVM::LLVMType>()
                                    : LLVM::LLVMType::getVoidTy(llvmDialect);

    SmallVector<LLVM::LLVMType, 4> argTypes;
    argTypes.reserve(type.getNumInputs());
    SmallVector<unsigned, 4> promotedArgIndices;
    promotedArgIndices.reserve(type.getNumInputs());

    // Convert the original function arguments. Struct arguments are promoted to
    // pointer to struct arguments to allow calling external functions with
    // various ABIs (e.g. compiled from C/C++ on platform X).
    auto varargsAttr = funcOp.getAttrOfType<BoolAttr>("std.varargs");
    TypeConverter::SignatureConversion result(funcOp.getNumArguments());
    for (auto en : llvm::enumerate(type.getInputs())) {
      auto t = en.value();
      auto converted = converter->convertType(t).dyn_cast<LLVM::LLVMType>();
      if (!converted)
        return matchFailure();
      if (t.isa<MemRefType>()) {
        converted = converted.getPointerTo();
        promotedArgIndices.push_back(en.index());
      }
      argTypes.push_back(converted);
    }
    for (unsigned idx = 0, e = argTypes.size(); idx < e; ++idx)
      result.addInputs(idx, argTypes[idx]);

    auto llvmType = LLVM::LLVMType::getFunctionTy(
        resultType, argTypes, varargsAttr && varargsAttr.getValue());

    // Only retain those attributes that are not constructed by build.
    SmallVector<NamedAttribute, 4> attributes;
    for (const auto &attr : funcOp.getAttrs()) {
      if (attr.first.is(SymbolTable::getSymbolAttrName()) ||
          attr.first.is(impl::getTypeAttrName()) ||
          attr.first.is("std.varargs"))
        continue;
      attributes.push_back(attr);
    }

    // Create an LLVM funcion, use external linkage by default until MLIR
    // functions have linkage.
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        op->getLoc(), funcOp.getName(), llvmType, LLVM::Linkage::External,
        attributes);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newFuncOp.getBody(), result);

    // Insert loads from memref descriptor pointers in function bodies.
    if (!newFuncOp.getBody().empty()) {
      Block *firstBlock = &newFuncOp.getBody().front();
      rewriter.setInsertionPoint(firstBlock, firstBlock->begin());
      for (unsigned idx : promotedArgIndices) {
        BlockArgument arg = firstBlock->getArgument(idx);
        Value loaded = rewriter.create<LLVM::LoadOp>(funcOp.getLoc(), arg);
        rewriter.replaceUsesOfBlockArgument(arg, loaded);
      }
    }

    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct FCReturnOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;
  FCTypeConverter *converter;

public:
  explicit FCReturnOpLowering(MLIRContext *_context, FCTypeConverter *converter)
      : ConversionPattern(FC::FCReturnOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        converter(converter) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numArguments = op->getNumOperands();

    // If ReturnOp has 0 or 1 operand, create it and return immediately.
    if (numArguments == 0) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, llvm::ArrayRef<Value>(),
                                                  llvm::ArrayRef<Block *>(),
                                                  op->getAttrs());
      return matchSuccess();
    }
    if (numArguments == 1) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
          op, llvm::ArrayRef<Value>(operands.front()),
          llvm::ArrayRef<Block *>(), op->getAttrs());
      return matchSuccess();
    }

    // Otherwise, we need to pack the arguments into an LLVM struct type before
    // returning.
    auto packedType = converter->packFunctionResults(
        llvm::to_vector<4>(op->getOperandTypes()));

    Value packed = rewriter.create<LLVM::UndefOp>(op->getLoc(), packedType);
    for (unsigned i = 0; i < numArguments; ++i) {
      packed = rewriter.create<LLVM::InsertValueOp>(
          op->getLoc(), packedType, packed, operands[i],
          rewriter.getI64ArrayAttr(i));
    }
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, llvm::makeArrayRef(packed),
                                                llvm::ArrayRef<Block *>(),
                                                op->getAttrs());
    return matchSuccess();
  }
};

// A CallOp automatically promotes MemRefType to a sequence of alloca /
// store and
// passes the pointer to the MemRef across function boundaries.
struct FCCallOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;
  FCTypeConverter *converter;

public:
  explicit FCCallOpLowering(MLIRContext *_context, FCTypeConverter *converter)
      : ConversionPattern(FC::FCCallOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        converter(converter) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<FC::FCCallOp> transformed(operands);
    auto callOp = cast<FC::FCCallOp>(op);

    // Pack the result types into a struct.
    Type packedResult;
    unsigned numResults = callOp.getNumResults();
    auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());
    if (numResults != 0) {
      if (!(packedResult = this->converter->packFunctionResults(resultTypes)))
        return this->matchFailure();
    }

    SmallVector<Value, 4> opOperands(op->getOperands());
    auto promoted = this->converter->promoteMemRefDescriptors(
        op->getLoc(), opOperands, operands, rewriter);
    auto newOp = rewriter.create<LLVM::CallOp>(op->getLoc(), packedResult,
                                               promoted, op->getAttrs());

    // If < 2 results, packing did not do anything and we can just return.
    if (numResults < 2) {
      SmallVector<Value, 4> results(newOp.getResults());
      rewriter.replaceOp(op, results);
      return this->matchSuccess();
    }

    // Otherwise, it had been converted to an operation producing a structure.
    // Extract individual results from the structure and return them as list.
    // TODO(aminim, ntv, riverriddle, zinenko): this seems like patching around
    // a particular interaction between MemRefType and CallOp lowering. Find a
    // way to avoid special casing.
    SmallVector<Value, 4> results;
    results.reserve(numResults);
    for (unsigned i = 0; i < numResults; ++i) {
      auto type = this->converter->convertType(op->getResult(i).getType());
      results.push_back(rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), type, newOp.getOperation()->getResult(0),
          rewriter.getIndexArrayAttr(i)));
    }
    rewriter.replaceOp(op, results);

    return this->matchSuccess();
  }
};

void fcmlir::populateFCFuncOpLoweringPatterns(
    OwningRewritePatternList &patterns, FCTypeConverter *typeConverter,
    MLIRContext *context) {
  patterns.insert<FCFuncOpLowering>(context, typeConverter);
  patterns.insert<FCReturnOpLowering>(context, typeConverter);
  patterns.insert<FCCallOpLowering>(context, typeConverter);
}