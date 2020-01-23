// Copyright (c) 2019, Compiler Tree Technologies Pvt Ltd.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "FCToLLVM/FCRuntimeHelper.h"
#include "FCToLLVM/FCToLLVMLowering.h"
#include "dialect/FC/FCOps.h"
#include "dialect/OpenMP/OpenMPOps.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#define PASS_NAME "FCToLLVMLowering"
#define DEBUG_TYPE PASS_NAME
// TODO: Add DEBUG WITH TYPE and STATISTIC

using namespace fcmlir;
using namespace mlir;
using namespace FC;

struct ComplexConstantOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;
  FCTypeConverter *converter;

public:
  explicit ComplexConstantOpLowering(MLIRContext *_context,
                                     FCTypeConverter *converter)
      : ConversionPattern(ComplexConstantOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto stringOp = cast<FC::ComplexConstantOp>(op);
    auto type = converter->convertType(stringOp.getResult().getType());
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, type, stringOp.value());
    return matchSuccess();
  }
};

class ComplexDescriptor {
private:
  // Complex type value in LLVM Dialect.
  mlir::Value val;
  mlir::Type complexEleType;
  ConversionPatternRewriter &builder;

public:
  explicit ComplexDescriptor(mlir::Value transformedVal,
                             ConversionPatternRewriter &builder)
      : val(transformedVal), builder(builder) {
    auto type = val.getType().cast<LLVM::LLVMType>();
    complexEleType = type.getArrayElementType();
    assert(complexEleType);
  }

  mlir::Value real() {
    return builder.create<LLVM::ExtractValueOp>(
        val.getLoc(), complexEleType, val, builder.getI64ArrayAttr(0));
  }

  mlir::Value complex() {
    return builder.create<LLVM::ExtractValueOp>(
        val.getLoc(), complexEleType, val, builder.getI64ArrayAttr(1));
  }

  mlir::Value setReal(mlir::Value real) {
    val = builder.create<LLVM::InsertValueOp>(val.getLoc(), val.getType(), val,
                                              real, builder.getI64ArrayAttr(0));
    return val;
  }

  mlir::Value setComplex(mlir::Value complex) {
    val = builder.create<LLVM::InsertValueOp>(
        val.getLoc(), val.getType(), val, complex, builder.getI64ArrayAttr(1));
    return val;
  }

  /// Builds IR creating an `undef` value of the descriptor type.
  static ComplexDescriptor undef(ConversionPatternRewriter &builder,
                                 Location loc, Type descriptorType) {
    Value descriptor = builder.create<LLVM::UndefOp>(
        loc, descriptorType.cast<LLVM::LLVMType>());
    return ComplexDescriptor(descriptor, builder);
  }
  mlir::Value value() { return val; }
};

struct ComplexAddOpLowering : public ConversionPattern {
  MLIRContext *context;
  FCTypeConverter *converter;

public:
  explicit ComplexAddOpLowering(MLIRContext *_context,
                                FCTypeConverter *converter)
      : ConversionPattern(ComplexAddOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto stringOp = cast<FC::ComplexAddOp>(op);
    auto type = converter->convertType(stringOp.getResult().getType());
    OperandAdaptor<FC::ComplexAddOp> transformed(operands);
    auto lhs = ComplexDescriptor(transformed.lhs(), rewriter);
    auto rhs = ComplexDescriptor(transformed.rhs(), rewriter);
    auto result = ComplexDescriptor::undef(rewriter, op->getLoc(), type);
    auto real =
        rewriter.create<LLVM::FAddOp>(op->getLoc(), lhs.real(), rhs.real());
    auto complex = rewriter.create<LLVM::FAddOp>(op->getLoc(), lhs.complex(),
                                                 rhs.complex());
    result.setReal(real);
    result.setComplex(complex);
    rewriter.replaceOp(op, result.value());
    return matchSuccess();
  }
};

struct ComplexSubOpLowering : public ConversionPattern {
  MLIRContext *context;
  FCTypeConverter *converter;

public:
  explicit ComplexSubOpLowering(MLIRContext *_context,
                                FCTypeConverter *converter)
      : ConversionPattern(ComplexSubOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto stringOp = cast<FC::ComplexSubOp>(op);
    auto type = converter->convertType(stringOp.getResult().getType());
    OperandAdaptor<FC::ComplexSubOp> transformed(operands);
    auto lhs = ComplexDescriptor(transformed.lhs(), rewriter);
    auto rhs = ComplexDescriptor(transformed.rhs(), rewriter);
    auto result = ComplexDescriptor::undef(rewriter, op->getLoc(), type);
    auto real =
        rewriter.create<LLVM::FSubOp>(op->getLoc(), lhs.real(), rhs.real());
    auto complex = rewriter.create<LLVM::FSubOp>(op->getLoc(), lhs.complex(),
                                                 rhs.complex());
    result.setReal(real);
    result.setComplex(complex);
    rewriter.replaceOp(op, result.value());
    return matchSuccess();
  }
};

struct ComplexMulOpLowering : public ConversionPattern {
  MLIRContext *context;
  FCTypeConverter *converter;

public:
  explicit ComplexMulOpLowering(MLIRContext *_context,
                                FCTypeConverter *converter)
      : ConversionPattern(FC::ComplexMulOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto stringOp = cast<FC::ComplexMulOp>(op);
    auto type = converter->convertType(stringOp.getResult().getType());
    OperandAdaptor<FC::ComplexMulOp> transformed(operands);
    auto lhs = ComplexDescriptor(transformed.lhs(), rewriter);
    auto rhs = ComplexDescriptor(transformed.rhs(), rewriter);
    auto x = lhs.real();
    auto y = lhs.complex();
    auto u = rhs.real();
    auto v = rhs.complex();

    // Real part.
    auto xu = rewriter.create<LLVM::FMulOp>(op->getLoc(), x, u);
    auto yv = rewriter.create<LLVM::FMulOp>(op->getLoc(), y, v);
    auto real = rewriter.create<LLVM::FSubOp>(op->getLoc(), xu, yv);

    // Complex part.
    auto xv = rewriter.create<LLVM::FMulOp>(op->getLoc(), x, v);
    auto yu = rewriter.create<LLVM::FMulOp>(op->getLoc(), y, u);
    auto complex = rewriter.create<LLVM::FAddOp>(op->getLoc(), xv, yu);

    auto result = ComplexDescriptor::undef(rewriter, op->getLoc(), type);
    result.setReal(real);
    result.setComplex(complex);
    rewriter.replaceOp(op, result.value());
    return matchSuccess();
  }
};

struct ComplexDivOpLowering : public ConversionPattern {
  MLIRContext *context;
  FCTypeConverter *converter;

public:
  explicit ComplexDivOpLowering(MLIRContext *_context,
                                FCTypeConverter *converter)
      : ConversionPattern(FC::ComplexDivOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto stringOp = cast<FC::ComplexDivOp>(op);
    auto type = converter->convertType(stringOp.getResult().getType());
    OperandAdaptor<FC::ComplexDivOp> transformed(operands);
    auto lhs = ComplexDescriptor(transformed.lhs(), rewriter);
    auto rhs = ComplexDescriptor(transformed.rhs(), rewriter);
    auto a = lhs.real();
    auto b = lhs.complex();
    auto c = rhs.real();
    auto d = rhs.complex();

    auto loc = op->getLoc();
    auto c2 = rewriter.create<LLVM::FMulOp>(loc, c, c);
    auto d2 = rewriter.create<LLVM::FMulOp>(loc, d, d);
    auto denom = rewriter.create<LLVM::FAddOp>(loc, c2, d2);

    // Real part.
    auto ac = rewriter.create<LLVM::FMulOp>(op->getLoc(), a, c);
    auto bd = rewriter.create<LLVM::FMulOp>(op->getLoc(), b, d);
    auto num = rewriter.create<LLVM::FAddOp>(op->getLoc(), ac, bd);
    auto real = rewriter.create<LLVM::FDivOp>(loc, num, denom);

    // Complex part.
    auto bc = rewriter.create<LLVM::FMulOp>(op->getLoc(), b, c);
    auto ad = rewriter.create<LLVM::FMulOp>(op->getLoc(), a, d);
    auto num2 = rewriter.create<LLVM::FSubOp>(op->getLoc(), bc, ad);
    auto complex = rewriter.create<LLVM::FDivOp>(loc, num2, denom);

    auto result = ComplexDescriptor::undef(rewriter, op->getLoc(), type);
    result.setReal(real);
    result.setComplex(complex);
    rewriter.replaceOp(op, result.value());
    return matchSuccess();
  }
};

struct ComplexConjugateOpLowering : public ConversionPattern {
  MLIRContext *context;
  FCTypeConverter *converter;

public:
  explicit ComplexConjugateOpLowering(MLIRContext *_context,
                                      FCTypeConverter *converter)
      : ConversionPattern(ComplexConjugateOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto stringOp = cast<FC::ComplexConjugateOp>(op);
    auto type = converter->convertType(stringOp.getResult().getType());
    auto val = ComplexDescriptor(operands[0], rewriter);
    auto result = ComplexDescriptor::undef(rewriter, op->getLoc(), type);
    auto complexVal = val.complex();
    auto eleTy = complexVal.getType().cast<LLVM::LLVMType>();
    mlir::Attribute value;
    if (eleTy.isFloatTy()) {
      value = rewriter.getF32FloatAttr(0);
    } else if (eleTy.isDoubleTy()) {
      value = rewriter.getF64FloatAttr(0);
    } else {
      assert(false);
    }
    auto realConstant =
        rewriter.create<LLVM::ConstantOp>(op->getLoc(), eleTy, value);
    auto complex =
        rewriter.create<LLVM::FSubOp>(op->getLoc(), realConstant, complexVal);
    auto real = val.real();

    result.setReal(real);
    result.setComplex(complex);
    rewriter.replaceOp(op, result.value());
    return matchSuccess();
  }
};

void fcmlir::populateComplexOpLoweringPatterns(
    OwningRewritePatternList &patterns, FCTypeConverter *typeConverter,
    MLIRContext *context) {
  patterns.insert<ComplexConstantOpLowering>(context, typeConverter);
  patterns.insert<ComplexAddOpLowering>(context, typeConverter);
  patterns.insert<ComplexSubOpLowering>(context, typeConverter);
  patterns.insert<ComplexMulOpLowering>(context, typeConverter);
  patterns.insert<ComplexDivOpLowering>(context, typeConverter);
  patterns.insert<ComplexConjugateOpLowering>(context, typeConverter);
}
