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
//===- FCToLLVMLowering.cpp - Loop.for to affine.for conversion -----------===//
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of mos to FC dialect operations to LLVM dialect
//
// NOTE: This code contains lot of repeated code. Needs lot of cleanup/
// refactoring.
// TODO: Split patterns into multiple files liek ArrayOps patterns, etc..
//===----------------------------------------------------------------------===//

#include "FCToLLVM/FCToLLVMLowering.h"
#include "FCToLLVM/FCRuntimeHelper.h"
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

static int counter = 0;

// NOTE: Code copied from ConvertStandardToLLVM.cpp and modified.

// template <typename Elem, size_t Rank>
// struct {
//   Elem *allocatedPtr;  // field 0
//   int64_t offset; // field 1
//   int64_t sizes[Rank] // field 2
//   int64_t strides[Rank];// field 3
//   int64_t lowerBounds[Rank]; // field 4
//   int64_t upperBounds[Rank];// field 5
// };
// An `alloc` is converted into a definition of a array descriptor value and
// a call to `malloc` to allocate the underlying data buffer.  The memref
// descriptor is of the LLVM structure type where the first element is a
// pointer to the (typed) data buffer, and the remaining elements serve to
// store dynamic sizes of the memref using LLVM-converted `index` type.

/// Helper class to produce LLVM dialect operations extracting or inserting
/// elements of a MemRef descriptor. Wraps a Value pointing to the
/// descriptor. The Value may be null, in which case none of the operations
/// are valid.
class ArrayDescriptor {
  using OpBuilder = mlir::OpBuilder;
  static constexpr unsigned kAllocPtr = 0;
  static constexpr unsigned kOffset = 1;
  static constexpr unsigned ksize = 2;
  static constexpr unsigned kstride = 3;
  static constexpr unsigned klb = 4;
  static constexpr unsigned kub = 5;

public:
  /// Construct a helper for the given descriptor value.
  explicit ArrayDescriptor(Value descriptor) : value(descriptor) {
    if (value) {
      structType = value.getType().cast<LLVM::LLVMType>();
      indexType =
          value.getType().cast<LLVM::LLVMType>().getStructElementType(kOffset);
    }
  }

  /// Builds IR creating an `undef` value of the descriptor type.
  static ArrayDescriptor undef(OpBuilder &builder, Location loc,
                               Type descriptorType) {
    Value descriptor = builder.create<LLVM::UndefOp>(
        loc, descriptorType.cast<LLVM::LLVMType>());
    return ArrayDescriptor(descriptor);
  }

  /// Builds IR extracting the allocated pointer from the descriptor.
  Value allocatedPtr(OpBuilder &builder, Location loc) {
    return extractPtr(builder, loc, kAllocPtr);
  }
  /// Builds IR inserting the allocated pointer into the descriptor.
  void setAllocatedPtr(OpBuilder &builder, Location loc, Value ptr) {
    setPtr(builder, loc, kAllocPtr, ptr);
  }

  /// Builds IR extracting the offset from the descriptor.
  Value offset(OpBuilder &builder, Location loc) {
    return builder.create<LLVM::ExtractValueOp>(
        loc, indexType, value, builder.getI64ArrayAttr(kOffset));
  }
  /// Builds IR inserting the offset into the descriptor.
  void setOffset(OpBuilder &builder, Location loc, Value offset) {
    value = builder.create<LLVM::InsertValueOp>(
        loc, structType, value, offset, builder.getI64ArrayAttr(kOffset));
  }

  /// Builds IR extracting the pos-th size from the descriptor.
  Value size(OpBuilder &builder, Location loc, unsigned pos) {
    return builder.create<LLVM::ExtractValueOp>(
        loc, indexType, value, builder.getI64ArrayAttr({ksize, pos}));
  }
  /// Builds IR inserting the pos-th size into the descriptor
  void setSize(OpBuilder &builder, Location loc, unsigned pos, Value size) {
    value = builder.create<LLVM::InsertValueOp>(
        loc, structType, value, size, builder.getI64ArrayAttr({ksize, pos}));
  }

  /// Builds IR extracting the pos-th size from the descriptor.
  Value stride(OpBuilder &builder, Location loc, unsigned pos) {
    return builder.create<LLVM::ExtractValueOp>(
        loc, indexType, value, builder.getI64ArrayAttr({kstride, pos}));
  }

  /// Builds IR inserting the pos-th stride into the descriptor
  void setStride(OpBuilder &builder, Location loc, unsigned pos, Value stride) {
    value = builder.create<LLVM::InsertValueOp>(
        loc, structType, value, stride,
        builder.getI64ArrayAttr({kstride, pos}));
  }

  /// Builds IR extracting the pos-th LB from the descriptor.
  Value lowerBound(OpBuilder &builder, Location loc, unsigned pos) {
    return builder.create<LLVM::ExtractValueOp>(
        loc, indexType, value, builder.getI64ArrayAttr({klb, pos}));
  }

  /// Builds IR inserting the pos-th LB into the descriptor
  void setlowerBound(OpBuilder &builder, Location loc, unsigned pos, Value lb) {
    value = builder.create<LLVM::InsertValueOp>(
        loc, structType, value, lb, builder.getI64ArrayAttr({klb, pos}));
  }

  /// Builds IR extracting the pos-th UB from the descriptor.
  Value upperBound(OpBuilder &builder, Location loc, unsigned pos) {
    return builder.create<LLVM::ExtractValueOp>(
        loc, indexType, value, builder.getI64ArrayAttr({kub, pos}));
  }

  /// Builds IR inserting the pos-th UB into the descriptor
  void setupperBound(OpBuilder &builder, Location loc, unsigned pos, Value ub) {
    value = builder.create<LLVM::InsertValueOp>(
        loc, structType, value, ub, builder.getI64ArrayAttr({kub, pos}));
  }

  /// Returns the (LLVM) type this descriptor points to.
  LLVM::LLVMType getElementType() {
    return value.getType().cast<LLVM::LLVMType>().getStructElementType(
        kAllocPtr);
  }

  /*implicit*/ operator Value() { return value; }

private:
  Value extractPtr(OpBuilder &builder, Location loc, unsigned pos) {
    Type type = structType.cast<LLVM::LLVMType>().getStructElementType(pos);
    return builder.create<LLVM::ExtractValueOp>(loc, type, value,
                                                builder.getI64ArrayAttr(pos));
  }

  void setPtr(OpBuilder &builder, Location loc, unsigned pos, Value ptr) {
    value = builder.create<LLVM::InsertValueOp>(loc, structType, value, ptr,
                                                builder.getI64ArrayAttr(pos));
  }

  // Cached descriptor type.
  Type structType;

  // Cached index type.
  Type indexType;

  // Actual descriptor.
  Value value;
};

static LLVM::LLVMType getIndexType(LLVM::LLVMDialect *llvmDialect) {
  return LLVM::LLVMType::getIntNTy(
      llvmDialect,
      llvmDialect->getLLVMModule().getDataLayout().getPointerSizeInBits());
};

// Create an LLVM IR pseudo-operation defining the given index constant.
static mlir::Value createIndexConstant(ConversionPatternRewriter &builder,
                                       LLVM::LLVMDialect *llvmDialect,
                                       Location loc, uint64_t value) {
  auto attr = builder.getIntegerAttr(builder.getIndexType(), value);
  return builder.create<LLVM::ConstantOp>(loc, getIndexType(llvmDialect), attr);
};

static mlir::Value
getSizeFromArrayDescriptor(mlir::Value val, FC::ArrayType arrTy,
                           ConversionPatternRewriter &rewriter,
                           mlir::Location loc) {
  auto shape = arrTy.getShape();

  if (arrTy.hasStaticShape()) {
    int64_t size = 1;
    for (unsigned i = 0; i < shape.size(); ++i) {
      size = size * shape[i].size;
    }
    return rewriter.create<ConstantIndexOp>(loc, size);
  }
  auto rank = shape.size();
  ArrayDescriptor arrDescriptor(val);
  mlir::Value size = arrDescriptor.size(rewriter, loc, 0);
  for (unsigned I = 1; I < rank; ++I) {
    size = rewriter.create<mlir::MulIOp>(loc, size,
                                         arrDescriptor.size(rewriter, loc, I));
  }
  return size;
}

static LogicalResult getStridesAndOffset(FC::ArrayType t,
                                         SmallVectorImpl<int64_t> &strides,
                                         int64_t &offset) {

  auto numBounds = t.getRank();
  strides.resize(numBounds);
  if (!t.hasStaticShape()) {
    std::fill(strides.begin(), strides.end(),
              FC::ArrayType::getDynamicSizeValue());
    offset = FC::ArrayType::getDynamicSizeValue();
    return LogicalResult::Success;
  }

  auto shape = t.getShape();

  for (int j = 0; j < numBounds; ++j) {
    if (j == 0) {
      strides[j] = 1;
      offset = shape[j].lowerBound;
      continue;
    }
    strides[j] = strides[j - 1] * shape[j - 1].size;
    offset += strides[j] * shape[j].lowerBound;
  }
  offset = -offset;
  return LogicalResult::Success;
}

static mlir::Value
buildArrayDescriptorStruct(Operation *op, ArrayRef<mlir::Value> operands,
                           ConversionPatternRewriter &rewriter,
                           MLIRContext *context, LLVM::LLVMDialect *llvmDialect,
                           mlir::Value allocated = nullptr,
                           bool needMalloc = false) {
  auto loc = op->getLoc();
  assert(op->getNumResults() == 1);
  auto Ty = op->getResult(0).getType();
  Type eleTy = Ty;
  if (auto refType = Ty.dyn_cast<FC::RefType>())
    eleTy = refType.getEleTy();
  auto arrTy = eleTy.dyn_cast<FC::ArrayType>();
  FCTypeConverter lowering(context);
  int alignment = 0;

  // Get the MLIR type wrapping the LLVM i8* type.
  auto getVoidPtrType = [&]() {
    return LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  };

  // This is a scalar type. Return plain alloca type.
  if (!arrTy) {
    if (allocated) {
      auto zero = createIndexConstant(rewriter, llvmDialect, loc, 0);
      auto gep = rewriter.create<LLVM::GEPOp>(
          loc, allocated.getType(), ArrayRef<mlir::Value>{allocated, zero});
      return gep;
    }
    auto ty = lowering.convertType(eleTy);
    auto llTy = ty.cast<LLVM::LLVMType>();
    auto constant = createIndexConstant(rewriter, llvmDialect, loc, 1);
    auto alloca = rewriter.create<LLVM::AllocaOp>(loc, llTy.getPointerTo(),
                                                  constant, alignment);
    return alloca;
  }

  bool hasStaticShape = arrTy.hasStaticShape();
  // Get actual sizes of the memref as values
  //   : static sizes are constant
  // values and dynamic sizes are passed to 'alloc' as operands.  In
  // case of zero-dimensional memref, assume a scalar (size 1).

  llvm::SmallVector<mlir::Value, 2> strides, lowerBounds, upperBounds, sizes;
  mlir::Value offset;
  auto rank = arrTy.getRank();
  auto One = createIndexConstant(rewriter, llvmDialect, loc, 1);

  if (hasStaticShape) {
    for (auto &dim : arrTy.getShape())
      sizes.push_back(
          createIndexConstant(rewriter, llvmDialect, loc, dim.size));
  } else {
    for (unsigned j = 0; j < rank; ++j) {
      auto lower = operands[2 * j];
      lowerBounds.push_back(lower);
      auto upper = operands[2 * j + 1];
      upperBounds.push_back(upper);
      auto loc = upper.getLoc();
      auto size = rewriter.create<mlir::SubIOp>(loc, upper, lower).getResult();
      size = rewriter.create<mlir::AddIOp>(loc, size, One);
      sizes.push_back(size);
    }
  }

  // Compute the total number of memref elements.
  mlir::Value cumulativeSize = sizes.front();
  for (unsigned i = 1, e = sizes.size(); i < e; ++i)
    cumulativeSize = rewriter.create<LLVM::MulOp>(
        loc, getIndexType(llvmDialect),
        ArrayRef<mlir::Value>{cumulativeSize, sizes[i]});

  // Compute the size of an individual element. This emits the MLIR
  // equivalent of the following sizeof(...) implementation in LLVM IR:
  //   %0 = getelementptr %elementType* null, %indexType 1
  //   %1 = ptrtoint %elementType* %0 to %indexType
  // which is a common pattern of getting the size of a type in bytes.
  auto elementType = arrTy.getEleTy();
  auto convertedPtrType =
      lowering.convertType(elementType).cast<LLVM::LLVMType>().getPointerTo();
  auto nullPtr = rewriter.create<LLVM::NullOp>(loc, convertedPtrType);
  auto one = createIndexConstant(rewriter, llvmDialect, loc, 1);
  auto gep = rewriter.create<LLVM::GEPOp>(loc, convertedPtrType,
                                          ArrayRef<mlir::Value>{nullPtr, one});
  auto elementSize =
      rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(llvmDialect), gep);
  cumulativeSize = rewriter.create<LLVM::MulOp>(
      loc, getIndexType(llvmDialect),
      ArrayRef<mlir::Value>{cumulativeSize, elementSize});

  // Allocate the underlying buffer and store a pointer to it in the MemRef
  // descriptor.

  if (!allocated) {
    if (arrTy.hasStaticShape() && !needMalloc) {
      allocated = rewriter.create<LLVM::AllocaOp>(loc, getVoidPtrType(),
                                                  cumulativeSize, alignment);
    } else {
      // Insert the `malloc` declaration if it is not already present.
      auto module = op->getParentOfType<ModuleOp>();
      auto mallocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("malloc");
      if (!mallocFunc) {
        mlir::OpBuilder moduleBuilder(
            op->getParentOfType<ModuleOp>().getBodyRegion());
        mallocFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
            rewriter.getUnknownLoc(), "malloc",
            LLVM::LLVMType::getFunctionTy(getVoidPtrType(),
                                          getIndexType(llvmDialect),
                                          /*isVarArg=*/false));
      }

      allocated = rewriter
                      .create<LLVM::CallOp>(
                          loc, getVoidPtrType(),
                          rewriter.getSymbolRefAttr(mallocFunc), cumulativeSize)
                      .getResult(0);
    }
  }

  auto structElementType = lowering.convertType(elementType);
  auto elementPtrType = structElementType.cast<LLVM::LLVMType>().getPointerTo();
  mlir::Value bitcastAllocated = rewriter.create<LLVM::BitcastOp>(
      loc, elementPtrType, ArrayRef<mlir::Value>(allocated));

  // Create the MemRef descriptor.
  auto structType = lowering.convertType(arrTy);
  auto memRefDescriptor = ArrayDescriptor::undef(rewriter, loc, structType);
  // Field 1: Allocated pointer, used for malloc/free.
  memRefDescriptor.setAllocatedPtr(rewriter, loc, bitcastAllocated);

  if (arrTy.hasStaticShape()) {
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto shape = arrTy.getShape();
    auto successStrides = getStridesAndOffset(arrTy, strides, offset);
    assert(succeeded(successStrides) && "unexpected non-strided memref");
    (void)successStrides;
    // 0-D memref corner case: they have size 1 ...
    assert(((arrTy.getRank() == 0 && strides.empty() && sizes.size() == 1) ||
            (strides.size() == sizes.size())) &&
           "unexpected number of strides");

    // Field 2: offset
    memRefDescriptor.setOffset(
        rewriter, loc, createIndexConstant(rewriter, llvmDialect, loc, offset));

    // Field 3, 4, 5, 6: size and strides, lower and upper bounds
    for (auto indexedArg : llvm::enumerate(shape)) {
      auto index = indexedArg.index();
      auto size = createIndexConstant(rewriter, llvmDialect, loc,
                                      indexedArg.value().size);
      auto stride =
          createIndexConstant(rewriter, llvmDialect, loc, strides[index]);
      auto lb = createIndexConstant(rewriter, llvmDialect, loc,
                                    indexedArg.value().lowerBound);
      auto ub = createIndexConstant(rewriter, llvmDialect, loc,
                                    indexedArg.value().upperBound);
      memRefDescriptor.setSize(rewriter, loc, index, size);
      memRefDescriptor.setStride(rewriter, loc, index, stride);
      memRefDescriptor.setlowerBound(rewriter, loc, index, lb);
      memRefDescriptor.setupperBound(rewriter, loc, index, ub);
    }
    return memRefDescriptor;
  }

  // Collect dynamic bounds size and symbols.
  assert(rank * 2 == operands.size());

  strides.resize(rank);
  // Calculate Stride.
  for (int j = 0; j < rank; ++j) {
    if (j == 0) {
      strides[j] = One;
      offset = lowerBounds[j];
      continue;
    }
    strides[j] =
        rewriter.create<mlir::MulIOp>(loc, strides[j - 1], sizes[j - 1]);
    auto temp = rewriter.create<mlir::MulIOp>(loc, strides[j], lowerBounds[j]);
    offset = rewriter.create<mlir::AddIOp>(loc, offset, temp);
  }
  auto minusOne = rewriter.create<mlir::ConstantIndexOp>(loc, -1);
  offset = rewriter.create<mlir::MulIOp>(loc, minusOne, offset);

  // Field 2: offset
  memRefDescriptor.setOffset(rewriter, loc, offset);

  // Field 3, 4, 5, 6: size and strides, lower and upper bounds
  for (auto indexedArg : llvm::enumerate(strides)) {
    auto index = indexedArg.index();
    memRefDescriptor.setSize(rewriter, loc, index, sizes[index]);
    memRefDescriptor.setStride(rewriter, loc, index, strides[index]);
    memRefDescriptor.setlowerBound(rewriter, loc, index, lowerBounds[index]);
    memRefDescriptor.setupperBound(rewriter, loc, index, upperBounds[index]);
  }
  // Return the final value of the descriptor.
  return memRefDescriptor;
}

struct FCSprintfOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCSprintfOpLowering(MLIRContext *_context)
      : ConversionPattern(FC::SprintfOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);

    auto loc = op->getLoc();
    auto unit = operands[0];

    auto sprintfOp = cast<FC::SprintfOp>(op);
    const auto &stringAttr =
        sprintfOp.getAttr("arg_info").cast<FC::StringInfoAttr>();
    const auto &stringInfo = stringAttr.getStringInfo();

    ArrayDescriptor arrDesc(unit);
    unit = arrDesc.allocatedPtr(rewriter, loc);
    auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    unit = rewriter.create<mlir::LLVM::BitcastOp>(loc, I8Ptr, unit);

    llvm::SmallVector<mlir::Value, 2> args;

    std::map<unsigned, mlir::Value> arraySizeMap;
    std::map<unsigned, mlir::Type> arrayTypeMap;

    for (unsigned i = 1; i < operands.size(); ++i) {
      auto opType = operands[i].getType();
      if (auto refType = opType.dyn_cast<FC::RefType>()) {
        auto eleTy = refType.getEleTy();
        if (auto arrTy = eleTy.dyn_cast<FC::ArrayType>()) {
          arraySizeMap[i] =
              getSizeFromArrayDescriptor(operands[i], arrTy, rewriter, loc);
          arrayTypeMap[i] = arrTy.getEleTy();
        }
      } else {
        arraySizeMap[i] = nullptr;
        arrayTypeMap[i] = nullptr;
      }
    }

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    fc::RuntimeHelper *helper =
        new fc::RuntimeHelper(parentModule, context, llvmDialect);

    for (unsigned i = 1; i < operands.size(); ++i) {
      auto actualArg = operands[i];
      auto isString = stringInfo.test(i - 1);
      if (arraySizeMap[i] || isString) {
        ArrayDescriptor arrDesc(actualArg);
        actualArg = arrDesc.allocatedPtr(rewriter, loc);
      }
      helper->fillPrintArgsFor(actualArg, args, loc, arraySizeMap[i],
                               arrayTypeMap[i], &rewriter, false, isString);
    }

    args.insert(args.begin(), unit);
    args.insert(args.begin(),
                rewriter.create<ConstantIntOp>(loc, args.size() - 1,
                                               rewriter.getIntegerType(32)));

    auto sprintFn = helper->getSprintfFunction(rewriter);
    ArrayRef<mlir::Type> results;
    rewriter.create<CallOp>(loc, sprintFn, results, args);

    rewriter.eraseOp(op);

    return matchSuccess();
  }
};

struct FCUndefLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;
  FCTypeConverter *converter;

public:
  explicit FCUndefLowering(MLIRContext *_context, FCTypeConverter *converter)
      : ConversionPattern(FC::UndefOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(
        op, converter->convertType(op->getResult(0).getType()));
    return matchSuccess();
  }
};

struct GetPointerToOpLowering : public ConversionPattern {
  MLIRContext *context;

public:
  explicit GetPointerToOpLowering(MLIRContext *_context)
      : ConversionPattern(FC::GetPointerToOp::getOperationName(), 1, _context),
        context(_context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands[0]);
    return matchSuccess();
  }
};

struct FCItoSOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCItoSOpLowering(MLIRContext *_context)
      : ConversionPattern(FC::ItoSOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);

    auto loc = op->getLoc();
    auto dest = operands[0];
    auto src = operands[1];

    ArrayDescriptor arrDesc(dest);
    dest = arrDesc.allocatedPtr(rewriter, loc);
    auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    dest = rewriter.create<mlir::LLVM::BitcastOp>(loc, I8Ptr, dest);

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    fc::RuntimeHelper *helper =
        new fc::RuntimeHelper(parentModule, context, llvmDialect);
    llvm::SmallVector<mlir::Value, 2> args{dest, src};
    auto strCpyFn = helper->getIntToStringFunction(rewriter);

    ArrayRef<mlir::Type> results;
    rewriter.create<CallOp>(loc, strCpyFn, results, args);
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct FCStoIOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCStoIOpLowering(MLIRContext *_context)
      : ConversionPattern(FC::StoIOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);

    auto loc = op->getLoc();
    auto unit = operands[0];
    auto stoiOp = cast<FC::StoIOp>(op);

    if (llvm::isa<mlir::LLVM::InsertValueOp>(unit.getDefiningOp())) {
      ArrayDescriptor arrDesc(unit);
      unit = arrDesc.allocatedPtr(rewriter, loc);
    }
    auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    unit = rewriter.create<mlir::LLVM::BitcastOp>(loc, I8Ptr, unit);

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    fc::RuntimeHelper *helper =
        new fc::RuntimeHelper(parentModule, context, llvmDialect);

    llvm::SmallVector<mlir::Value, 2> args{unit};
    auto stoiFn = helper->getStringToIntFunction(rewriter);

    ArrayRef<mlir::Type> results{mlir::LLVM::LLVMType::getInt32Ty(llvmDialect)};
    auto callOp = rewriter.create<CallOp>(loc, stoiFn, results, args);
    stoiOp.getResult().replaceAllUsesWith(callOp.getResult(0));
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct FCStrCatOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCStrCatOpLowering(MLIRContext *_context)
      : ConversionPattern(FC::StrCatOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);

    auto loc = op->getLoc();
    auto lhs = operands[0];
    auto rhs = operands[1];
    auto strcatOp = cast<FC::StrCatOp>(op);

    ArrayDescriptor lhsArrDesc(lhs);
    lhs = lhsArrDesc.allocatedPtr(rewriter, loc);

    ArrayDescriptor rhsArrDesc(rhs);
    rhs = rhsArrDesc.allocatedPtr(rewriter, loc);

    auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    lhs = rewriter.create<mlir::LLVM::BitcastOp>(loc, I8Ptr, lhs);
    rhs = rewriter.create<mlir::LLVM::BitcastOp>(loc, I8Ptr, rhs);

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    fc::RuntimeHelper *helper =
        new fc::RuntimeHelper(parentModule, context, llvmDialect);
    llvm::SmallVector<mlir::Value, 2> args{lhs, rhs};
    auto strCatFn = helper->getStrCatFunction(rewriter);

    ArrayRef<mlir::Type> results{I8Ptr};
    auto callOp = rewriter.create<CallOp>(loc, strCatFn, results, args);
    strcatOp.getResult().replaceAllUsesWith(callOp.getResult(0));
    rewriter.eraseOp(op);

    return matchSuccess();
  }
};

struct FCTrimOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCTrimOpLowering(MLIRContext *_context)
      : ConversionPattern(FC::TrimOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    OperandAdaptor<FC::TrimOp> transformed(operands);

    auto loc = op->getLoc();
    auto src = transformed.str();
    ArrayDescriptor arrayDesc(src);
    src = arrayDesc.allocatedPtr(rewriter, loc);

    llvm::SmallVector<mlir::Type, 2> types(op->getOperandTypes());
    auto opType = types[0];
    if (auto refType = opType.dyn_cast<FC::RefType>()) {
      opType = refType.getEleTy();
    }

    auto arrTy = opType.dyn_cast<FC::ArrayType>();
    assert(arrTy);
    auto size =
        getSizeFromArrayDescriptor(transformed.str(), arrTy, rewriter, loc);

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    fc::RuntimeHelper *helper =
        new fc::RuntimeHelper(parentModule, context, llvmDialect);
    llvm::SmallVector<mlir::Value, 2> args{src, size};
    auto trimFn = helper->getTrimFunction(rewriter);

    ArrayRef<mlir::Type> results;
    rewriter.create<CallOp>(loc, trimFn, results, args);
    rewriter.eraseOp(op);

    return matchSuccess();
  }
};

struct FCStrCpyOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCStrCpyOpLowering(MLIRContext *_context)
      : ConversionPattern(FC::StrCpyOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);

    auto loc = op->getLoc();
    auto dest = operands[0];
    auto src = operands[1];

    // TODO CLEANUP use the extractor
    if (auto insertValue = llvm::dyn_cast<mlir::LLVM::InsertValueOp>(
            operands[0].getDefiningOp())) {
      while (auto v = llvm::dyn_cast<mlir::LLVM::InsertValueOp>(
                 insertValue.container().getDefiningOp())) {
        insertValue = v;
      }

      dest = insertValue.value();
    }

    if (auto insertValue = llvm::dyn_cast<mlir::LLVM::InsertValueOp>(
            operands[1].getDefiningOp())) {
      while (auto v = llvm::dyn_cast<mlir::LLVM::InsertValueOp>(
                 insertValue.container().getDefiningOp())) {
        insertValue = v;
      }

      src = insertValue.value();
    }

    auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    src = rewriter.create<mlir::LLVM::BitcastOp>(loc, I8Ptr, src);
    dest = rewriter.create<mlir::LLVM::BitcastOp>(loc, I8Ptr, dest);

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    fc::RuntimeHelper *helper =
        new fc::RuntimeHelper(parentModule, context, llvmDialect);
    llvm::SmallVector<mlir::Value, 2> args{dest, src};
    auto strCpyFn = helper->getStrCpyFunction(rewriter);

    ArrayRef<mlir::Type> results;
    rewriter.create<CallOp>(loc, strCpyFn, results, args);
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct FCStoIAOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCStoIAOpLowering(MLIRContext *_context)
      : ConversionPattern(FC::StoIAOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);

    auto loc = op->getLoc();
    auto unitExpr = operands[0];
    auto exprVal = operands[1];

    // TODO : CLEANUP
    //        Seems like duplicate code
    if (auto insertValue = llvm::dyn_cast_or_null<mlir::LLVM::InsertValueOp>(
            operands[0].getDefiningOp())) {
      while (auto v = llvm::dyn_cast<mlir::LLVM::InsertValueOp>(
                 insertValue.container().getDefiningOp())) {
        insertValue = v;
      }

      unitExpr = insertValue.value();
    }

    if (auto insertValue = llvm::dyn_cast_or_null<mlir::LLVM::InsertValueOp>(
            operands[1].getDefiningOp())) {
      while (auto v = llvm::dyn_cast<mlir::LLVM::InsertValueOp>(
                 insertValue.container().getDefiningOp())) {
        insertValue = v;
      }

      exprVal = insertValue.value();
    }

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    fc::RuntimeHelper *helper =
        new fc::RuntimeHelper(parentModule, context, llvmDialect);

    auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    auto I32Ptr = mlir::LLVM::LLVMType::getInt32Ty(llvmDialect).getPointerTo();

    ArrayDescriptor exprValDesc(exprVal);
    unitExpr = rewriter.create<mlir::LLVM::BitcastOp>(loc, I8Ptr, unitExpr);
    exprVal = rewriter.create<mlir::LLVM::BitcastOp>(
        loc, I32Ptr, exprValDesc.allocatedPtr(rewriter, loc));

    llvm::SmallVector<mlir::Value, 2> args{unitExpr, exprVal};
    auto stoiaFn = helper->getStoIAFunction(rewriter);

    ArrayRef<mlir::Type> results;
    rewriter.create<CallOp>(loc, stoiaFn, results, args);
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct FCLowerBoundOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCLowerBoundOpLowering(MLIRContext *_context)
      : ConversionPattern(FC::LBoundOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto lbOp = cast<FC::LBoundOp>(op);
    ArrayDescriptor aDescriptor(operands[0]);
    auto lbound = aDescriptor.lowerBound(rewriter, loc, lbOp.getDim() - 1);
    lbOp.getResult().replaceAllUsesWith(lbound);
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct FCUpperBoundOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCUpperBoundOpLowering(MLIRContext *_context)
      : ConversionPattern(FC::UBoundOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto loc = op->getLoc();
    auto ubOp = cast<FC::UBoundOp>(op);
    ArrayDescriptor aDescriptor(operands[0]);
    auto lbound = aDescriptor.upperBound(rewriter, loc, ubOp.getDim() - 1);
    ubOp.getResult().replaceAllUsesWith(lbound);
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct FCCastOpLowering : public ConversionPattern {
private:
  FCTypeConverter *converter;

  int getNumBits(mlir::LLVM::LLVMType type) const {
    if (type.isIntegerTy(1))
      return 1;
    if (type.isIntegerTy(8))
      return 8;
    if (type.isIntegerTy(16))
      return 16;
    if (type.isIntegerTy(32))
      return 32;
    if (type.isIntegerTy(64))
      return 64;
    if (type.isIntegerTy(128))
      return 128;
    llvm_unreachable("Unhandled");
  }

public:
  explicit FCCastOpLowering(MLIRContext *_context, FCTypeConverter *converter)
      : ConversionPattern(CastOp::getOperationName(), 1, _context),
        converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);

    auto castOp = cast<FC::CastOp>(op);
    auto loc = op->getLoc();
    auto type = castOp.getResult().getType();
    auto fromTy =
        converter->convertType(castOp.value().getType()).cast<LLVM::LLVMType>();
    auto toTy = converter->convertType(type).cast<LLVM::LLVMType>();

    mlir::Value finalResult = nullptr;

    // TODO cleanup: implement it using switch(type.getKind());
    if (fromTy.isIntegerTy() && (toTy.isFloatTy() || toTy.isDoubleTy())) {
      auto siToFp =
          rewriter.create<mlir::LLVM::SIToFPOp>(loc, toTy, castOp.value());
      finalResult = siToFp.res();
    } else if (fromTy.isIntegerTy() && toTy.isIntegerTy()) {
      if (getNumBits(fromTy) < getNumBits(toTy)) {
        auto signExt =
            rewriter.create<mlir::LLVM::SExtOp>(loc, toTy, castOp.value());
        finalResult = signExt.res();
      } else if (getNumBits(fromTy) > getNumBits(toTy)) {
        auto trunc =
            rewriter.create<mlir::LLVM::TruncOp>(loc, toTy, castOp.value());
        finalResult = trunc.res();

      } else if (getNumBits(fromTy) == getNumBits(toTy)) {
        finalResult = castOp.value();
      } else {
        llvm_unreachable("should not reach here");
      }
    } else if (toTy.isIntegerTy() &&
               (fromTy.isFloatTy() || fromTy.isDoubleTy())) {
      auto fptosi =
          rewriter.create<mlir::LLVM::FPToSIOp>(loc, toTy, castOp.value());
      finalResult = fptosi.res();
    } else if (fromTy.isFloatTy() && toTy.isDoubleTy()) {
      auto fpExt =
          rewriter.create<mlir::LLVM::FPExtOp>(loc, toTy, castOp.value());
      finalResult = fpExt.res();
    } else if (toTy.isFloatTy() && fromTy.isDoubleTy()) {
      auto fpTrunc =
          rewriter.create<mlir::LLVM::FPTruncOp>(loc, toTy, castOp.value());
      finalResult = fpTrunc.res();
    } else if (toTy == fromTy) {
      // Nothing to do. it is all the same.
      finalResult = castOp.value();
    } else {
      llvm::errs() << toTy << "\n";
      llvm::errs() << fromTy << "\n";
      llvm_unreachable("Unhandled castop lowering");
    }

    castOp.getResult().replaceAllUsesWith(finalResult);
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct FCCastToMemRefOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;
  FCTypeConverter *lowering;

public:
  explicit FCCastToMemRefOpLowering(MLIRContext *_context,
                                    FCTypeConverter *_lowering)
      : ConversionPattern(FC::CastToMemRefOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        lowering(_lowering) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto castOp = cast<FC::CastToMemRefOp>(op);

    mlir::Value memRef = castOp.getResult();

    auto memRefType = memRef.getType().cast<MemRefType>();

    auto structType = lowering->convertType(memRefType);
    auto memRefDesc = MemRefDescriptor::undef(rewriter, loc, structType);
    ArrayDescriptor arrDesc(operands[0]); // ie. fcRef in llvm-dialect

    // set allocated-ptr
    auto alloc = arrDesc.allocatedPtr(rewriter, loc);
    memRefDesc.setAllocatedPtr(rewriter, loc, alloc);
    memRefDesc.setAlignedPtr(rewriter, loc, alloc);

    // set offset
    auto offset = arrDesc.offset(rewriter, loc);
    memRefDesc.setOffset(rewriter, loc, offset);

    // set size and stride for each dimension
    llvm::ArrayRef<int64_t> shape = memRefType.getShape();
    for (unsigned pos = 0; pos < shape.size(); ++pos) {
      auto size = arrDesc.size(rewriter, loc, pos);
      auto stride = arrDesc.stride(rewriter, loc, pos);
      memRefDesc.setSize(rewriter, loc, pos, size);
      memRefDesc.setStride(rewriter, loc, pos, stride);
    }

    rewriter.replaceOp(op, {memRefDesc});
    return matchSuccess();
  }
};

class FCWriteOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCWriteOpLowering(MLIRContext *_context)
      : ConversionPattern(WriteOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    fc::RuntimeHelper helper(parentModule, context, llvmDialect);
    mlir::Location loc = op->getLoc();
    auto writeOp = cast<FC::WriteOp>(op);
    auto stringAttr = writeOp.getAttr("arg_info").cast<FC::StringInfoAttr>();
    auto spaceAttr = writeOp.getAttr("space_list").cast<mlir::ArrayAttr>();
    mlir::SymbolRefAttr writeFn = helper.getFileWriteFunction(rewriter);

    rewriter.setInsertionPoint(op);

    llvm::SmallVector<mlir::Value, 2> args;

    std::map<unsigned, mlir::Value> arraySizeMap;
    std::map<unsigned, mlir::Type> arrayTypeMap;

    SmallVector<int32_t, 2> spaceList;
    for (auto &n : spaceAttr.getValue()) {
      spaceList.push_back(n.cast<IntegerAttr>().getValue().getSExtValue());
    }

    for (auto arg : llvm::enumerate(op->getOperandTypes())) {
      auto i = arg.index();
      auto opType = arg.value();
      if (auto refType = opType.dyn_cast<FC::RefType>()) {
        opType = refType.getEleTy();
      }
      if (auto arrTy = opType.dyn_cast<FC::ArrayType>()) {
        arraySizeMap[i] =
            getSizeFromArrayDescriptor(operands[i], arrTy, rewriter, loc);
        arrayTypeMap[i] = arrTy.getEleTy();
      } else {
        arraySizeMap[i] = nullptr;
        arrayTypeMap[i] = nullptr;
      }
    }

    unsigned k = 0;
    for (unsigned i = 0; i < operands.size() - 1; i++) {
      mlir::Value iarg = operands[i + 1]; // Skip the unit
      if (arraySizeMap[i + 1]) {
        ArrayDescriptor arrDesc(iarg);
        iarg = arrDesc.allocatedPtr(rewriter, loc);
      }
      // push default numspaces for now
      bool isString = stringAttr.getStringInfo().test(i);
      mlir::Value numSpaces = nullptr;
      numSpaces = rewriter.create<ConstantIntOp>(loc, spaceList[k++],
                                                 rewriter.getIntegerType(32));
      args.push_back(numSpaces);

      helper.fillPrintArgsFor(iarg, args, loc, arraySizeMap[i + 1],
                              arrayTypeMap[i + 1], &rewriter, false, isString);
    }

    args.insert(args.begin(),
                rewriter.create<ConstantIntOp>(loc, args.size(),
                                               rewriter.getIntegerType(32)));
    args.insert(args.begin(), operands[0]); // Push the unit

    ArrayRef<mlir::Type> results;
    rewriter.create<CallOp>(loc, writeFn, results, args);

    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

class FCReadOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCReadOpLowering(MLIRContext *_context)
      : ConversionPattern(ReadOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    fc::RuntimeHelper helper(parentModule, context, llvmDialect);
    mlir::Location loc = op->getLoc();

    auto readOp = cast<FC::ReadOp>(op);
    auto stringAttr = readOp.getAttr("arg_info").cast<FC::StringInfoAttr>();

    mlir::SymbolRefAttr readFn = helper.getFileReadFunction(rewriter);

    rewriter.setInsertionPoint(op);

    llvm::SmallVector<mlir::Value, 2> args;

    std::map<unsigned, mlir::Value> arraySizeMap;
    std::map<unsigned, mlir::Type> arrayTypeMap;

    for (auto arg : llvm::enumerate(op->getOperandTypes())) {
      auto i = arg.index();
      auto opType = arg.value();
      if (auto refType = opType.dyn_cast<FC::RefType>()) {
        opType = refType.getEleTy();
      }
      if (auto arrTy = opType.dyn_cast<FC::ArrayType>()) {
        arraySizeMap[i] =
            getSizeFromArrayDescriptor(operands[i], arrTy, rewriter, loc);
        arrayTypeMap[i] = arrTy.getEleTy();
      } else {
        arraySizeMap[i] = nullptr;
        arrayTypeMap[i] = nullptr;
      }
    }

    for (unsigned i = 0; i < operands.size() - 1; i++) {
      mlir::Value iarg = operands[i + 1]; // Skip the unit

      if (arraySizeMap[i + 1]) {
        ArrayDescriptor arrDesc(iarg);
        iarg = arrDesc.allocatedPtr(rewriter, loc);
      }
      bool isString = stringAttr.getStringInfo().test(i);
      helper.fillReadArgsFor(iarg, args, loc, arraySizeMap[i + 1],
                             arrayTypeMap[i + 1], &rewriter, isString);
    }

    args.insert(args.begin(),
                rewriter.create<ConstantIntOp>(loc, args.size(),
                                               rewriter.getIntegerType(32)));
    args.insert(args.begin(), operands[0]); // Push the unit

    ArrayRef<mlir::Type> results{mlir::LLVM::LLVMType::getInt32Ty(llvmDialect)};
    auto call = rewriter.create<CallOp>(loc, readFn, results, args);
    readOp.getResult().replaceAllUsesWith(call.getResult(0));
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct FCStringOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCStringOpLowering(MLIRContext *_context)
      : ConversionPattern(StringOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto stringOp = cast<FC::StringOp>(op);
    auto value = stringOp.value().str() + '\0';
    mlir::Value stringValue = mlir::LLVM::createGlobalString(
        op->getLoc(), rewriter, "str_const_" + std::to_string(counter), value,
        mlir::LLVM::Linkage::Internal, llvmDialect);
    auto returnVal = buildArrayDescriptorStruct(
        op, {}, rewriter, context, llvmDialect, stringValue, false);
    rewriter.replaceOp(op, returnVal);
    counter++;
    return matchSuccess();
  }
};

// TODO: Currently, the memref type is hardcoded to the memref descriptor
// structure. No way to get the custom type we need.
static LLVM::LLVMType createLLVMTypeFor(FCTypeConverter &converter,
                                        FC::RefType type) {
  auto baseTy = type.getEleTy();
  // Nested memrefs are not handled!
  assert(!baseTy.isa<FC::RefType>());
  auto arrType = baseTy.dyn_cast<FC::ArrayType>();

  if (!arrType) {
    return converter.convertType(baseTy).cast<LLVM::LLVMType>();
  }
  baseTy = arrType.getEleTy();

  auto llBaseTy = converter.convertType(baseTy).cast<LLVM::LLVMType>();
  auto shape = arrType.getShape();

  // FIXME: Assuming the simple strided column major layout map.
  int64_t totalSize = 1;
  for (auto val : shape) {
    assert(val.size != -1);
    totalSize *= val.size;
  }
  return LLVM::LLVMType::getArrayTy(llBaseTy, totalSize);
}

struct FCCloseOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCCloseOpLowering(MLIRContext *_context)
      : ConversionPattern(CloseOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    fc::RuntimeHelper *helper =
        new fc::RuntimeHelper(parentModule, context, llvmDialect);

    auto closeOp = cast<FC::CloseOp>(op);
    auto loc = op->getLoc();
    rewriter.setInsertionPoint(op);
    llvm::SmallVector<mlir::Value, 2> args;
    args.push_back(closeOp.unit());
    auto closeFn = helper->getCloseFunction(rewriter);
    ArrayRef<mlir::Type> results{mlir::LLVM::LLVMType::getInt32Ty(llvmDialect)};

    auto call = rewriter.create<CallOp>(loc, closeFn, results, args);
    closeOp.getResult().replaceAllUsesWith(call.getResult(0));
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct FCOpenOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCOpenOpLowering(MLIRContext *_context)
      : ConversionPattern(OpenOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    fc::RuntimeHelper *helper =
        new fc::RuntimeHelper(parentModule, context, llvmDialect);

    auto openOp = cast<FC::OpenOp>(op);
    auto loc = op->getLoc();
    auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    rewriter.setInsertionPoint(op);

    auto file = openOp.file();
    auto type = file.getType();
    if (auto refType = type.dyn_cast<FC::RefType>()) {
      type = refType.getEleTy();
    }
    OperandAdaptor<FC::OpenOp> transformed(operands);
    if (type.isa<FC::ArrayType>()) {
      ArrayDescriptor desc(transformed.file());
      file = rewriter.create<mlir::LLVM::BitcastOp>(
          loc, I8Ptr, desc.allocatedPtr(rewriter, loc));
    }
    llvm::SmallVector<mlir::Value, 2> args;
    args.push_back(openOp.unit());
    args.push_back(file);
    auto openFn = helper->getOpenFunction(rewriter);

    ArrayRef<mlir::Type> results{mlir::LLVM::LLVMType::getInt32Ty(llvmDialect)};
    auto call = rewriter.create<CallOp>(loc, openFn, results, args);
    openOp.getResult().replaceAllUsesWith(call.getResult(0));
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct FCPrintOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;
  fc::RuntimeHelper &helper;

public:
  explicit FCPrintOpLowering(MLIRContext *_context, fc::RuntimeHelper &helper)
      : ConversionPattern(PrintOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        helper(helper) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op->getLoc();
    auto printOp = cast<FC::PrintOp>(op);
    const auto &stringAttr =
        printOp.getAttr("arg_info").cast<FC::StringInfoAttr>();
    const auto &stringInfo = stringAttr.getStringInfo();
    auto printFn = helper.getPrintFunction(rewriter);

    rewriter.setInsertionPoint(op);

    llvm::SmallVector<mlir::Value, 2> args;

    std::map<unsigned, mlir::Value> arraySizeMap;
    std::map<unsigned, mlir::Type> arrayTypeMap;

    for (auto arg : llvm::enumerate(op->getOperandTypes())) {
      auto i = arg.index();
      auto opType = arg.value();
      if (auto refType = opType.dyn_cast<FC::RefType>()) {
        opType = refType.getEleTy();
      }
      if (auto arrTy = opType.dyn_cast<FC::ArrayType>()) {
        arraySizeMap[i] =
            getSizeFromArrayDescriptor(operands[i], arrTy, rewriter, loc);
        arrayTypeMap[i] = arrTy.getEleTy();
      } else {
        arraySizeMap[i] = nullptr;
        arrayTypeMap[i] = nullptr;
      }
    }
    for (auto arg : llvm::enumerate(operands)) {
      auto i = arg.index();
      auto actualArg = arg.value();
      if (arraySizeMap[i]) {
        ArrayDescriptor arrDesc(actualArg);
        actualArg = arrDesc.allocatedPtr(rewriter, loc);
      }
      auto isString = stringInfo.test(i);
      helper.fillPrintArgsFor(actualArg, args, loc, arraySizeMap[i],
                              arrayTypeMap[i], &rewriter, false, isString);
    }

    args.insert(args.begin(),
                rewriter.create<ConstantIntOp>(loc, args.size(),
                                               rewriter.getIntegerType(32)));

    ArrayRef<mlir::Type> results;
    rewriter.replaceOpWithNewOp<CallOp>(op, printFn, results, args);
    return matchSuccess();
  }
};

struct FCArrayCmpIOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

  LLVM::LLVMFuncOp getStrCmpFunc(mlir::ModuleOp module,
                                 ConversionPatternRewriter &rewriter) const {
    auto memcpyFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("strcmp");
    auto I8PtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
    if (memcpyFunc) {
      return memcpyFunc;
    }
    mlir::OpBuilder moduleBuilder(module.getBodyRegion());
    auto int32 = LLVM::LLVMType::getInt32Ty(llvmDialect);
    auto funcTy = LLVM::LLVMType::getFunctionTy(int32, {I8PtrTy, I8PtrTy},
                                                /*isVarArg=*/false);
    memcpyFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), "strcmp", funcTy);
    return memcpyFunc;
  }

  LLVM::CallOp emitStrCmpCall(FC::ArrayCmpIOp cmpOp,
                              ArrayRef<mlir::Value> operands,
                              ConversionPatternRewriter &rewriter) const {
    OperandAdaptor<FC::ArrayCmpIOp> transformed(operands);
    ArrayDescriptor lhs(transformed.lhs());
    ArrayDescriptor rhs(transformed.rhs());
    // Do a memcopy from the value to pointer location.
    auto loc = cmpOp.getLoc();
    auto lhsPtr = lhs.allocatedPtr(rewriter, loc);
    auto rhsPtr = rhs.allocatedPtr(rewriter, loc);

    auto voidPtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
    auto lhsVoidPtr = rewriter.create<LLVM::BitcastOp>(loc, voidPtrTy, lhsPtr);
    auto rhsVoidPtr = rewriter.create<LLVM::BitcastOp>(loc, voidPtrTy, rhsPtr);

    auto module = cmpOp.getParentOfType<mlir::ModuleOp>();
    auto strcmpFunc = getStrCmpFunc(module, rewriter);

    llvm::SmallVector<mlir::Value, 2> args({lhsVoidPtr, rhsVoidPtr});

    auto callVal = rewriter.create<LLVM::CallOp>(
        loc, LLVM::LLVMType::getInt32Ty(llvmDialect),
        rewriter.getSymbolRefAttr(strcmpFunc), args);
    return callVal;
  }

public:
  explicit FCArrayCmpIOpLowering(MLIRContext *_context)
      : ConversionPattern(FC::ArrayCmpIOp ::getOperationName(),
                          PatternBenefit::impossibleToMatch(), _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto cmpOp = llvm::cast<FC::ArrayCmpIOp>(op);
    auto predicate = cmpOp.getPredicate();
    assert(predicate == CmpIPredicate::eq || predicate == CmpIPredicate::ne);
    auto strCmp = emitStrCmpCall(cmpOp, operands, rewriter);
    auto Int32 = LLVM::LLVMType::getInt32Ty(llvmDialect);
    auto attr = rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0);
    auto Zero = rewriter.create<LLVM::ConstantOp>(cmpOp.getLoc(), Int32, attr);

    LLVM::ICmpPredicate llvmPred;
    // TODO : move to a function.
    switch (predicate) {
    case CmpIPredicate::eq:
      llvmPred = LLVM::ICmpPredicate::eq;
      break;
    case CmpIPredicate::ne:
      llvmPred = LLVM::ICmpPredicate::ne;
      break;
    default:
      llvm_unreachable("Unhandled MLIR to LLVM CmpIPred translation");
    };
    auto returnVal = rewriter.create<LLVM::ICmpOp>(cmpOp.getLoc(), llvmPred,
                                                   strCmp.getResult(0), Zero);
    rewriter.replaceOp(op, returnVal.getResult());
    return matchSuccess();
  }
};

struct FCAllocaOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCAllocaOpLowering(MLIRContext *_context)
      : ConversionPattern(FC::AllocaOp ::getOperationName(),
                          PatternBenefit::impossibleToMatch(), _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto allocOp = llvm::cast<FC::AllocaOp>(op);
    bool needMalloc = false;
    if (auto attr =
            allocOp.getAttr("use_malloc").dyn_cast_or_null<BoolAttr>()) {
      needMalloc = attr.getValue();
    }
    auto returnVal = buildArrayDescriptorStruct(
        op, operands, rewriter, context, llvmDialect, nullptr, needMalloc);
    assert(returnVal);
    rewriter.replaceOp(op, returnVal);
    return matchSuccess();
  }
};

// A `dealloc` is converted into a call to `free` on the underlying data
// buffer. The memref descriptor being an SSA value, there is no need to
// clean it up in any way.
struct FCDeallocaOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCDeallocaOpLowering(MLIRContext *_context)
      : ConversionPattern(FC::DeallocaOp ::getOperationName(),
                          PatternBenefit::impossibleToMatch(), _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    // Get the MLIR type wrapping the LLVM i8* type.
    auto getVoidPtrType = [&]() {
      return LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    };

    auto getVoidType = [&]() { return LLVM::LLVMType::getVoidTy(llvmDialect); };

    assert(operands.size() == 1 && "dealloc takes one operand");
    OperandAdaptor<DeallocOp> transformed(operands);

    // Insert the `free` declaration if it is not already present.
    auto freeFunc =
        op->getParentOfType<ModuleOp>().lookupSymbol<LLVM::LLVMFuncOp>("free");
    if (!freeFunc) {
      mlir::OpBuilder moduleBuilder(
          op->getParentOfType<ModuleOp>().getBodyRegion());
      freeFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), "free",
          LLVM::LLVMType::getFunctionTy(getVoidType(), getVoidPtrType(),
                                        /*isVarArg=*/false));
    }

    ArrayDescriptor memref(transformed.memref());
    Value casted = rewriter.create<LLVM::BitcastOp>(
        op->getLoc(), getVoidPtrType(),
        memref.allocatedPtr(rewriter, op->getLoc()));
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, ArrayRef<Type>(), rewriter.getSymbolRefAttr(freeFunc), casted);

    /*
     * This will not work, we need some kind of global map
    auto deAllocOp = cast<FC::DeallocaOp>(op);
    auto loc = op->getLoc();
    auto zero = rewriter.create<ConstantIntOp>(loc, 0,
                                          rewriter.getIntegerType(64));
    if (auto refType =
    deAllocOp.ref().getType().dyn_cast<FC::RefType>()) { if (auto arrTy
    = refType.getEleTy().dyn_cast<FC::ArrayType>()) { auto rank =
    arrTy.getShape().size(); for (unsigned i = 0; i < rank; i++) {
          memref.setSize(rewriter, loc, i, zero);
          //memref.setlowerBound(rewriter, loc, i, one);
          //memref.setupperBound(rewriter, loc, i, zero);
        }
      }
    }
    */
    return matchSuccess();
  }
};

struct FCGlobalOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit FCGlobalOpLowering(MLIRContext *_context)
      : ConversionPattern(GlobalOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto globalOp = cast<FC::GlobalOp>(op);
    FCTypeConverter typeConverter(context);

    auto memref = globalOp.getType();
    bool isDoublePtr = false;
    if (auto refType1 = memref.getEleTy().dyn_cast<FC::RefType>()) {
      if (auto refType2 = refType1.getEleTy().dyn_cast<FC::RefType>()) {
        memref = refType2;
        isDoublePtr = true;
      }
    }
    auto isStatic = memref.isStatic();

    // Create a global plain type of llvm and then wrap it with the
    // memref descriptor.

    if (isStatic) {
      auto ActualTy = createLLVMTypeFor(typeConverter, memref);
      auto llActualTy = ActualTy.cast<LLVM::LLVMType>();
      if (isDoublePtr)
        llActualTy = llActualTy.getPointerTo().getPointerTo();

      // TODO : Make it work for extern.
      auto actualGlobal = rewriter.create<mlir::LLVM::GlobalOp>(
          globalOp.getLoc(), llActualTy, false, mlir::LLVM::Linkage::Internal,
          globalOp.sym_name().str(), globalOp.getValueOrNull());
      assert(actualGlobal);
    } else {
      mlir::LLVM::LLVMType llTy = typeConverter.convertType(globalOp.getType())
                                      .cast<mlir::LLVM::LLVMType>();

      mlir::LLVM::GlobalOp llGlobal = rewriter.create<mlir::LLVM::GlobalOp>(
          globalOp.getLoc(), llTy, false, mlir::LLVM::Linkage::Internal,
          globalOp.sym_name().str(), globalOp.getValueOrNull());
      assert(llGlobal);
    }

    // If it is a dynamic memref type. Someone else will fill it.
    // Do not worry for now.
    rewriter.eraseOp(globalOp);
    counter++;
    return matchSuccess();
  }
};

struct FCAddrOfOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;
  mlir::ModuleOp &module;

public:
  explicit FCAddrOfOpLowering(mlir::ModuleOp &module, MLIRContext *_context)
      : ConversionPattern(AddressOfOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        module(module) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto addrOfOp = cast<FC::AddressOfOp>(op);
    FCTypeConverter typeConverter(context);

    auto global = module.lookupSymbol<LLVM::GlobalOp>(addrOfOp.global_name());
    if (!global) {
      return matchFailure();
    }

    mlir::LLVM::LLVMType llTy = typeConverter.convertType(global.getType())
                                    .cast<mlir::LLVM::LLVMType>();

    auto llOp = rewriter.create<mlir::LLVM::AddressOfOp>(
        addrOfOp.getLoc(), llTy.getPointerTo(), addrOfOp.global_name());

    auto type = addrOfOp.getType().cast<FC::RefType>();
    if (auto refType1 = type.getEleTy().dyn_cast<FC::RefType>()) {
      if (auto refType2 = refType1.getEleTy().dyn_cast<FC::RefType>()) {
        type = refType2;
      }
    }

    mlir::Value returnVal = nullptr;
    if (type.isStatic()) {
      llvm::SmallVector<mlir::Value, 2> empty;
      returnVal = buildArrayDescriptorStruct(op, empty, rewriter, context,
                                             llvmDialect, llOp, false);
    } else {
      returnVal = rewriter.create<LLVM::LoadOp>(op->getLoc(), llOp);
    }
    assert(returnVal);
    rewriter.replaceOp(addrOfOp, {returnVal});
    counter++;
    return matchSuccess();
  }
};

// Convert the element type of the memref `t` to to an LLVM type using
// `lowering`, get a pointer LLVM type pointing to the converted `t`,
// wrap it into the MLIR LLVM dialect type and return.
static mlir::Type getMemRefElementPtrType(FC::RefType t,
                                          FCTypeConverter *lowering) {
  auto elementType = t.getEleTy().cast<FC::ArrayType>();
  auto converted = lowering->convertType(elementType.getEleTy());
  if (!converted)
    return {};
  return converted.cast<LLVM::LLVMType>().getPointerTo();
}

// Common base for load and store operations on MemRefs.  Restricts the
// match to supported MemRef types.  Provides functionality to emit code
// accessing a specific element of the underlying data buffer.
template <typename Derived>
struct LoadStoreOpLowering : public ConversionPattern {
protected:
  MLIRContext *context;
  FCTypeConverter *lowering;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit LoadStoreOpLowering(MLIRContext *_context, FCTypeConverter *lowering)
      : ConversionPattern(Derived::getOperationName(),
                          PatternBenefit::impossibleToMatch(), _context),
        context(_context), lowering(lowering),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  // Given subscript indices and array sizes in row-major order,
  //   i_n, i_{n-1}, ..., i_1
  //   s_n, s_{n-1}, ..., s_1
  // obtain a value that corresponds to the linearized subscript
  //   \sum_k i_k * \prod_{j=1}^{k-1} s_j
  // by accumulating the running linearized value.
  // Note that `indices` and `allocSizes` are passed in the same order
  // as they appear in load/store operations and memref type
  // declarations.
  mlir::Value linearizeSubscripts(ConversionPatternRewriter &builder,
                                  Location loc, ArrayRef<mlir::Value> indices,
                                  ArrayRef<mlir::Value> allocSizes) const {
    assert(indices.size() == allocSizes.size() &&
           "mismatching number of indices and allocation sizes");
    assert(!indices.empty() && "cannot linearize a 0-dimensional access");

    mlir::Value linearized = indices.front();
    for (int i = 1, nSizes = allocSizes.size(); i < nSizes; ++i) {
      linearized = builder.create<LLVM::MulOp>(
          loc, this->getIndexType(),
          ArrayRef<mlir::Value>{linearized, allocSizes[i]});
      linearized = builder.create<LLVM::AddOp>(
          loc, this->getIndexType(),
          ArrayRef<mlir::Value>{linearized, indices[i]});
    }
    return linearized;
  }

  // This is a strided getElementPtr variant that linearizes subscripts
  // as:
  //   `base_offset + index_0 * stride_0 + ... + index_n * stride_n`.
  mlir::Value getStridedElementPtr(Location loc, mlir::Type elementTypePtr,
                                   mlir::Value descriptor,
                                   ArrayRef<mlir::Value> indices,
                                   ArrayRef<int64_t> strides, int64_t offset,
                                   ConversionPatternRewriter &rewriter,
                                   bool IsDynamicOffset) const {
    ArrayDescriptor memRefDescriptor(descriptor);

    auto getIndexType = [&]() {
      return LLVM::LLVMType::getIntNTy(
          llvmDialect,
          llvmDialect->getLLVMModule().getDataLayout().getPointerSizeInBits());
    };

    // Create an LLVM IR pseudo-operation defining the given index
    // constant.
    auto createIndexConstant = [&](ConversionPatternRewriter &builder,
                                   Location loc, uint64_t value) {
      auto attr = builder.getIntegerAttr(builder.getIndexType(), value);
      return builder.create<LLVM::ConstantOp>(loc, getIndexType(), attr);
    };

    mlir::Value base = memRefDescriptor.allocatedPtr(rewriter, loc);
    mlir::Value offsetValue = IsDynamicOffset
                                  ? memRefDescriptor.offset(rewriter, loc)
                                  : createIndexConstant(rewriter, loc, offset);

    for (int i = 0, e = indices.size(); i < e; ++i) {
      mlir::Value stride = strides[i] == FC::ArrayType::getDynamicSizeValue()
                               ? memRefDescriptor.stride(rewriter, loc, i)
                               : createIndexConstant(rewriter, loc, strides[i]);
      mlir::Value additionalOffset =
          rewriter.create<LLVM::MulOp>(loc, indices[i], stride);
      offsetValue =
          rewriter.create<LLVM::AddOp>(loc, offsetValue, additionalOffset);
    }
    return rewriter.create<LLVM::GEPOp>(loc, elementTypePtr, base, offsetValue);
  }

  mlir::Value getDataPtr(Location loc, FC::RefType type, mlir::Value memRefDesc,
                         ArrayRef<mlir::Value> indices,
                         ConversionPatternRewriter &rewriter,
                         llvm::Module &module) const {
    auto ptrType = getMemRefElementPtrType(type, lowering);
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto arrTy = type.getEleTy().cast<FC::ArrayType>();
    auto successStrides = getStridesAndOffset(arrTy, strides, offset);
    assert(succeeded(successStrides) && "unexpected non-strided memref");
    (void)successStrides;
    return getStridedElementPtr(loc, ptrType, memRefDesc, indices, strides,
                                offset, rewriter, !arrTy.hasStaticShape());
  }
};

// Load operation is lowered to obtaining a pointer to the indexed
// element and loading it.
struct FCLoadOpLowering : public LoadStoreOpLowering<FC::FCLoadOp> {
public:
  explicit FCLoadOpLowering(MLIRContext *_context, FCTypeConverter *lowering)
      : LoadStoreOpLowering(_context, lowering) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loadOp = cast<FCLoadOp>(op);
    OperandAdaptor<FC::FCLoadOp> transformed(operands);

    auto eleTy = loadOp.getPointer().getType();
    if (auto refType = eleTy.dyn_cast<FC::RefType>()) {
      eleTy = refType.getEleTy();
    } else if (auto ptrType = eleTy.dyn_cast<FC::PointerType>()) {
      eleTy = ptrType.getEleTy();
    }

    mlir::Value dataPtr = nullptr;

    // Handle array loads.
    if (loadOp.getType().isa<FC::ArrayType>()) {
      assert(loadOp.indices().empty() && "Array section not handled yet");
      rewriter.replaceOp(op, transformed.pointer());
      return matchSuccess();
    }

    if (!eleTy.isa<FC::ArrayType>()) {
      // Scalar type.
      dataPtr = operands[0];
    } else {
      dataPtr = getDataPtr(op->getLoc(),
                           loadOp.pointer().getType().cast<FC::RefType>(),
                           transformed.pointer(), transformed.indices(),
                           rewriter, llvmDialect->getLLVMModule());
    }
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, dataPtr);
    return matchSuccess();
  }
};

struct FCArrayEleOpLowering : public LoadStoreOpLowering<FC::ArrayEleOp> {
public:
  explicit FCArrayEleOpLowering(MLIRContext *_context,
                                FCTypeConverter *lowering)
      : LoadStoreOpLowering(_context, lowering) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayEle = cast<ArrayEleOp>(op);
    OperandAdaptor<FC::ArrayEleOp> transformed(operands);
    auto type = arrayEle.getPointer().getType().cast<FC::RefType>();
    mlir::Value dataPtr = nullptr;

    if (!type.getEleTy().isa<FC::ArrayType>()) {
      dataPtr = operands[0];
    } else {
      dataPtr = getDataPtr(op->getLoc(), type, transformed.pointer(),
                           transformed.indices(), rewriter,
                           llvmDialect->getLLVMModule());
    }

    arrayEle.getResult().replaceAllUsesWith(dataPtr);
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

// Store operation is lowered to obtaining a pointer to the indexed
// element, and storing the given value to it.
struct FCStoreOpLowering : public LoadStoreOpLowering<FC::FCStoreOp> {
  LLVM::LLVMDialect *llvmDialect;

  LLVM::LLVMFuncOp getMemCopyFunc(mlir::ModuleOp module,
                                  ConversionPatternRewriter &rewriter) const {
    auto memcpyFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("memcpy");
    auto voidPtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
    if (memcpyFunc) {
      return memcpyFunc;
    }
    mlir::OpBuilder moduleBuilder(module.getBodyRegion());
    auto funcTy = LLVM::LLVMType::getFunctionTy(
        voidPtrTy, {voidPtrTy, voidPtrTy, getIndexType(llvmDialect)},
        /*isVarArg=*/false);
    memcpyFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), "memcpy", funcTy);
    return memcpyFunc;
  }

public:
  explicit FCStoreOpLowering(MLIRContext *_context, FCTypeConverter *lowering)
      : LoadStoreOpLowering(_context, lowering),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  handleArrayStore(FC::FCStoreOp op, ArrayRef<mlir::Value> operands,
                   ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    OperandAdaptor<FC::FCStoreOp> transformed(operands);
    ArrayDescriptor valueDesc(transformed.valueToStore());
    ArrayDescriptor pointerDesc(transformed.pointer());
    // Array sections are not handled yet.
    if (!op.indices().empty()) {
      llvm::SmallVector<mlir::Type, 2> types(op.getOperandTypes());
      auto opType = types[1];
      auto refType = opType.dyn_cast<FC::RefType>();
      assert(refType);
      opType = refType.getEleTy();

      auto arrTy = opType.dyn_cast<FC::ArrayType>();
      assert(arrTy);
      if (!arrTy.getEleTy().isInteger(8)) {
        llvm_unreachable("Unhandled partial array sections\n");
      }
      SmallVector<mlir::Value, 2> indices;
      SmallVector<mlir::Value, 2> oldIndices(op.indices());
      auto I64 = LLVM::LLVMType::getInt64Ty(llvmDialect);

      unsigned rank = arrTy.getShape().size();
      auto remaining = rank - oldIndices.size();
      for (unsigned i = 0; i < remaining; ++i) {
        indices.push_back(pointerDesc.lowerBound(rewriter, loc, i));
      }

      for (auto index : op.indices()) {
        index = rewriter.create<mlir::LLVM::BitcastOp>(loc, I64, index);
        indices.push_back(index);
      }

      auto sourcePtr = valueDesc.allocatedPtr(rewriter, loc);
      auto destPtr = getDataPtr(loc, refType, transformed.pointer(), indices,
                                rewriter, llvmDialect->getLLVMModule());

      auto valueType = types[0];
      auto valueArrTy = valueType.dyn_cast<FC::ArrayType>();
      assert(valueArrTy);
      auto size = getSizeFromArrayDescriptor(transformed.valueToStore(),
                                             valueArrTy, rewriter, loc);
      auto module = op.getParentOfType<mlir::ModuleOp>();
      auto memcpyFunc = getMemCopyFunc(module, rewriter);

      llvm::SmallVector<mlir::Value, 4> args({destPtr, sourcePtr, size});

      auto voidPtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
      auto callVal = rewriter.create<LLVM::CallOp>(
          loc, voidPtrTy, rewriter.getSymbolRefAttr(memcpyFunc), args);
      assert(callVal);

      rewriter.eraseOp(op);
      return matchSuccess();
    }

    assert(op.indices().empty() && "array sections are not handled yet.");

    // Do a memcopy from the value to pointer location.
    auto sourcePtr = valueDesc.allocatedPtr(rewriter, loc);
    auto destPtr = pointerDesc.allocatedPtr(rewriter, loc);
    auto srcType = op.valueToStore().getType().cast<FC::ArrayType>();
    auto dstEleType = op.pointer().getType().cast<FC::RefType>().getEleTy();
    auto dstType = dstEleType.cast<FC::ArrayType>();
    assert(srcType.getEleTy() == dstType.getEleTy());
    auto srcSize = getSizeFromArrayDescriptor(transformed.valueToStore(),
                                              srcType, rewriter, loc);
    auto dstSize = getSizeFromArrayDescriptor(transformed.pointer(), dstType,
                                              rewriter, loc);
    auto elemSize = srcType.getEleTy().getIntOrFloatBitWidth() / 8;

    // TODO: String copy can copy to a smaller size buffer. In that case
    // we need to get the minimum size of both arrays and copy!
    // FIXME; do this only for character array types?
    auto minSize = rewriter.create<mlir::CmpIOp>(loc, CmpIPredicate::slt,
                                                 srcSize, dstSize);
    auto sizeToCopy =
        rewriter.create<mlir::SelectOp>(loc, minSize, srcSize, dstSize);
    auto elemSizeVal =
        createIndexConstant(rewriter, llvmDialect, loc, elemSize);
    auto sizeInBytes =
        rewriter.create<mlir::MulIOp>(loc, sizeToCopy, elemSizeVal);

    auto voidPtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
    auto srcVoidPtr =
        rewriter.create<LLVM::BitcastOp>(loc, voidPtrTy, sourcePtr);
    auto dstVoidPtr = rewriter.create<LLVM::BitcastOp>(loc, voidPtrTy, destPtr);

    auto module = op.getParentOfType<mlir::ModuleOp>();
    auto memcpyFunc = getMemCopyFunc(module, rewriter);

    llvm::SmallVector<mlir::Value, 4> args(
        {dstVoidPtr, srcVoidPtr, sizeInBytes});

    auto callVal = rewriter.create<LLVM::CallOp>(
        loc, voidPtrTy, rewriter.getSymbolRefAttr(memcpyFunc), args);
    assert(callVal);

    rewriter.eraseOp(op);
    return matchSuccess();
  }

  bool handleStringStore(FC::FCStoreOp op, ArrayRef<mlir::Value> operands,
                         ConversionPatternRewriter &rewriter) const {

    OperandAdaptor<FC::FCStoreOp> transformed(operands);
    ArrayDescriptor pointerDesc(transformed.pointer());
    // Array sections are not handled yet.
    assert(op.indices().empty() && "array sections are not handled yet.");

    // Do a memcopy from the value to pointer location.
    auto loc = op.getLoc();
    auto srcPtr = transformed.valueToStore();
    auto destPtr = pointerDesc.allocatedPtr(rewriter, loc);

    auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    if (destPtr.getType() != I8Ptr)
      return false;

    ModuleOp parentModule = op.getParentOfType<ModuleOp>();
    fc::RuntimeHelper *helper =
        new fc::RuntimeHelper(parentModule, context, llvmDialect);
    llvm::SmallVector<mlir::Value, 2> args{destPtr, srcPtr};
    auto strCpyFn = helper->getStrCpyFunction(rewriter);

    ArrayRef<mlir::Type> results;
    rewriter.create<CallOp>(loc, strCpyFn, results, args);
    rewriter.eraseOp(op);
    return true;
  }

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto storeOp = cast<FCStoreOp>(op);
    OperandAdaptor<FC::FCStoreOp> transformed(operands);
    auto type = storeOp.getPointer().getType().dyn_cast<FC::RefType>();
    auto valueToStore = storeOp.getValueToStore();
    if (valueToStore.getType().isa<FC::ArrayType>()) {
      return handleArrayStore(storeOp, operands, rewriter);
    }

    auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    if (valueToStore.getType() == I8Ptr &&
        handleStringStore(storeOp, operands, rewriter)) {
      return matchSuccess();
    }

    mlir::Value dataPtr = nullptr;
    if (!type) {
      dataPtr = storeOp.getPointer();
    } else if (!type.getEleTy().isa<FC::ArrayType>()) {
      dataPtr = transformed.pointer();
    } else {
      dataPtr = getDataPtr(op->getLoc(), type, transformed.pointer(),
                           transformed.indices(), rewriter,
                           llvmDialect->getLLVMModule());
    }
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, transformed.valueToStore(),
                                               dataPtr);
    assert(dataPtr);
    return matchSuccess();
  }
};

struct FCArgcOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;
  mlir::ModuleOp &module;

public:
  explicit FCArgcOpLowering(MLIRContext *_context, mlir::ModuleOp &mod)
      : ConversionPattern(ArgcOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        module(mod) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto name = "fc.internal.argc";
    auto global = module.lookupSymbol<LLVM::GlobalOp>(name);
    auto loc = op->getLoc();
    if (!global) {
      mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      auto I32 = mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
      auto linkage = LLVM::Linkage::AvailableExternally;
      global = rewriter.create<LLVM::GlobalOp>(loc, I32, /*isConstant=*/false,
                                               linkage, name, Attribute());
    }
    Value globalPtr = rewriter.create<LLVM::AddressOfOp>(op->getLoc(), global);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, globalPtr);
    return matchSuccess();
  }
};

static LLVM::LLVMFuncOp getMemCopyFunc(mlir::ModuleOp module,
                                       ConversionPatternRewriter &rewriter,
                                       LLVM::LLVMDialect *llvmDialect) {

  auto memcpyFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("memcpy");
  auto voidPtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
  if (memcpyFunc) {
    return memcpyFunc;
  }
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto funcTy = LLVM::LLVMType::getFunctionTy(
      voidPtrTy, {voidPtrTy, voidPtrTy, getIndexType(llvmDialect)},
      /*isVarArg=*/false);
  memcpyFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                      "memcpy", funcTy);
  return memcpyFunc;
}

struct FCArgvOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;
  mlir::ModuleOp &module;

public:
  explicit FCArgvOpLowering(MLIRContext *_context, mlir::ModuleOp &mod)
      : ConversionPattern(ArgvOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        module(mod) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto name = "fc.internal.argv";
    auto global = module.lookupSymbol<LLVM::GlobalOp>(name);
    auto loc = op->getLoc();
    if (!global) {
      mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      auto I8Ptr =
          mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect).getPointerTo();
      auto linkage = LLVM::Linkage::AvailableExternally;
      global = rewriter.create<LLVM::GlobalOp>(loc, I8Ptr, /*isConstant=*/false,
                                               linkage, name, Attribute());
    }
    auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    OperandAdaptor<FC::ArgvOp> transformed(operands);
    Value globalPtr = rewriter.create<LLVM::AddressOfOp>(op->getLoc(), global);
    /*
    globalPtr = rewriter.create<LLVM::BitcastOp>(
        loc, I8Ptr.getPointerTo(), ArrayRef<mlir::Value >(globalPtr));
    */
    auto argvBase = rewriter.create<LLVM::LoadOp>(loc, globalPtr);
    auto argvGEP =
        rewriter.create<LLVM::GEPOp>(loc, I8Ptr, argvBase, transformed.pos());
    auto finalPtr = rewriter.create<LLVM::LoadOp>(loc, argvGEP);
    llvm::SmallVector<mlir::Type, 2> types(op->getOperandTypes());
    auto opType = types[1];

    if (auto refType = opType.dyn_cast<FC::RefType>()) {
      opType = refType.getEleTy();
    }

    auto arrTy = opType.dyn_cast<FC::ArrayType>();
    assert(arrTy);

    auto size =
        getSizeFromArrayDescriptor(transformed.str(), arrTy, rewriter, loc);

    auto dest = transformed.str();
    ArrayDescriptor arrayDesc(dest);
    dest = arrayDesc.allocatedPtr(rewriter, loc);

    auto memcpyFunc = getMemCopyFunc(module, rewriter, llvmDialect);
    llvm::SmallVector<mlir::Value, 4> args({dest, finalPtr, size});

    auto voidPtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
    auto callVal = rewriter.create<LLVM::CallOp>(
        loc, voidPtrTy, rewriter.getSymbolRefAttr(memcpyFunc), args);
    assert(callVal);

    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct FCNullPointerOpLowering : public ConversionPattern {
  MLIRContext *context;
  FCTypeConverter *converter;

public:
  explicit FCNullPointerOpLowering(MLIRContext *_context,
                                   FCTypeConverter *converter)
      : ConversionPattern(NullPointerOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<LLVM::NullOp>(
        op, converter->convertType(op->getResult(0).getType()));
    return matchSuccess();
  }
};

struct CmpPointerEqualOpLowering : public ConversionPattern {
  MLIRContext *context;
  FCTypeConverter *converter;

public:
  explicit CmpPointerEqualOpLowering(MLIRContext *_context,
                                     FCTypeConverter *converter)
      : ConversionPattern(CmpPointerEqualOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<FC::CmpPointerEqualOp> transformed(operands);
    auto lhs = transformed.lhs();
    auto rhs = transformed.rhs();
    auto llvmDialect(context->getRegisteredDialect<LLVM::LLVMDialect>());
    auto I64 = LLVM::LLVMType::getInt64Ty(llvmDialect);
    auto lhsToInt = rewriter.create<LLVM::PtrToIntOp>(op->getLoc(), I64, lhs);
    auto rhsToInt = rewriter.create<LLVM::PtrToIntOp>(op->getLoc(), I64, rhs);
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq,
                                              lhsToInt, rhsToInt);
    return matchSuccess();
  }
};

struct FCToLLVMLowering : public ModulePass<FCToLLVMLowering> {
  virtual void runOnModule() {

    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<OMP::OpenMPDialect>();
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    OwningRewritePatternList patterns;
    FCTypeConverter typeConverter(&getContext());

    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    auto module = getModule();

    auto _context = &getContext();
    auto llvmDialect = _context->getRegisteredDialect<LLVM::LLVMDialect>();
    fc::RuntimeHelper helper(module, _context, llvmDialect);
    mlir::OpBuilder builder(_context);
    builder.setInsertionPoint(module);
    ConversionPatternRewriter rewriter(_context, &typeConverter);
    helper.getPrintFunction(rewriter);

    // Cast operations.
    patterns.insert<FCStoIOpLowering>(&getContext());
    patterns.insert<FCStoIAOpLowering>(&getContext());
    patterns.insert<FCItoSOpLowering>(&getContext());
    patterns.insert<FCStringOpLowering>(&getContext());
    patterns.insert<FCCastOpLowering>(&getContext(), &typeConverter);

    // String related operation
    patterns.insert<FCStrCatOpLowering>(&getContext());
    patterns.insert<FCStrCpyOpLowering>(&getContext());
    patterns.insert<FCTrimOpLowering>(&getContext());

    // Array related operations.
    patterns.insert<FCArrayEleOpLowering>(&getContext(), &typeConverter);
    patterns.insert<FCLowerBoundOpLowering>(&getContext());
    patterns.insert<FCUpperBoundOpLowering>(&getContext());
    patterns.insert<FCArrayCmpIOpLowering>(&getContext());

    // IO related operations.
    patterns.insert<FCSprintfOpLowering>(&getContext());
    patterns.insert<FCWriteOpLowering>(&getContext());
    patterns.insert<FCPrintOpLowering>(&getContext(), helper);
    patterns.insert<FCOpenOpLowering>(&getContext());
    patterns.insert<FCCloseOpLowering>(&getContext());
    patterns.insert<FCReadOpLowering>(&getContext());

    // Command line related
    patterns.insert<FCArgcOpLowering>(&getContext(), module);
    patterns.insert<FCArgvOpLowering>(&getContext(), module);

    // Memory related operations.
    patterns.insert<FCAllocaOpLowering>(&getContext());
    patterns.insert<FCDeallocaOpLowering>(&getContext());
    patterns.insert<FCGlobalOpLowering>(&getContext());
    patterns.insert<FCAddrOfOpLowering>(module, &getContext());
    patterns.insert<FCLoadOpLowering>(&getContext(), &typeConverter);
    patterns.insert<FCStoreOpLowering>(&getContext(), &typeConverter);
    patterns.insert<FCCastToMemRefOpLowering>(&getContext(), &typeConverter);

    // Command line related
    patterns.insert<FCArgcOpLowering>(&getContext(), module);
    patterns.insert<FCArgvOpLowering>(&getContext(), module);

    // Pointer related operations.
    patterns.insert<GetPointerToOpLowering>(&getContext());
    patterns.insert<CmpPointerEqualOpLowering>(&getContext(), &typeConverter);
    patterns.insert<FCNullPointerOpLowering>(&getContext(), &typeConverter);

    // undef operation
    patterns.insert<FCUndefLowering>(&getContext(), &typeConverter);

    // Convert FC function related patterns.
    populateFCFuncOpLoweringPatterns(patterns, &typeConverter, &getContext());

    // Convert complex type related patterns
    populateComplexOpLoweringPatterns(patterns, &typeConverter, &getContext());

    if (failed(applyFullConversion(module, target, patterns, &typeConverter)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createFCToLLVMLoweringPass() {
  return std::make_unique<FCToLLVMLowering>();
}
