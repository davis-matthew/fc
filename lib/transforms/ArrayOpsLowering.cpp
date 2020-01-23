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
//===- ArrayOpsLoweringPass.cpp -   ---------------------------------------===//
//
// Lower array operatoins into fc.do loops.
// TODO: ArraySectionExpander AST pass should be completely moved here.
//===----------------------------------------------------------------------===//

#include "dialect/FC/FCOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "common/Debug.h"

using namespace std;

namespace mlir {

static std::string getTempName() {
  static int tempVal = 0;
  return "temp" + std::to_string(tempVal++);
}

template <class ArrayBinOp, class BinOp>
struct ArrayBinOpLoweringPattern : public OpRewritePattern<ArrayBinOp> {
public:
  using OpRewritePattern<ArrayBinOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(ArrayBinOp doOp,
                                     PatternRewriter &rewriter) const override;
}; // struct ArrayAddOpLoweringPattern

template <class ArrayBinOp, class BinOp>
PatternMatchResult
ArrayBinOpLoweringPattern<ArrayBinOp, BinOp>::matchAndRewrite(
    ArrayBinOp arrayop, PatternRewriter &rewriter) const {

  auto mlirloc = arrayop.getLoc();
  auto step = rewriter.create<mlir::ConstantIntOp>(mlirloc, 1, 32);
  auto rank = arrayop.opd1().getType().template cast<FC::ArrayType>().getRank();
  auto attr = rewriter.getStringAttr("arrayop");

  // If opd is not a Load, allocate memory store to it.
  Value opd1mem, opd2mem;
  if (auto opd1load = dyn_cast<FC::FCLoadOp>(arrayop.opd1().getDefiningOp())) {
    opd1mem = opd1load.getPointer();
  } else {
    opd1mem = rewriter.create<FC::AllocaOp>(
        mlirloc, getTempName(), FC::RefType::get(arrayop.opd1().getType()));
    rewriter.create<FC::FCStoreOp>(mlirloc, arrayop.opd1(), opd1mem);
  }
  if (auto opd2load = dyn_cast<FC::FCLoadOp>(arrayop.opd2().getDefiningOp())) {
    opd2mem = opd2load.getPointer();
  } else {
    opd2mem = rewriter.create<FC::AllocaOp>(
        mlirloc, getTempName(), FC::RefType::get(arrayop.opd2().getType()));
    rewriter.create<FC::FCStoreOp>(mlirloc, arrayop.opd2(), opd2mem);
  }

  // Allocate memory for the result. If the only use of this op
  // is a store back of the array to memory, it can be done directly.
  FC::AllocaOp resultMem;
  bool createNewAlloc = true;
  if (arrayop.getResult().hasOneUse()) {
    for (auto u : arrayop.getResult().getUsers()) {
      if (auto so = dyn_cast<FC::FCStoreOp>(u)) {
        assert(so.getValueToStore() == arrayop);
        resultMem = cast<FC::AllocaOp>(so.getPointer().getDefiningOp());
        createNewAlloc = false;
        // The store itself is going to be dead, so delete it.
        rewriter.eraseOp(so);
      }
    }
  }
  if (createNewAlloc) {
    resultMem = rewriter.create<FC::AllocaOp>(
        mlirloc, getTempName(), FC::RefType::get(arrayop.getType()));
  }

  OpBuilder builder = rewriter;
  SmallVector<FC::SubscriptRange, 4> indices;
  for (int i = 0; i < rank; i++) {
    auto dim = builder.create<mlir::ConstantIndexOp>(mlirloc, i + 1);
    auto lb =
        builder
            .create<FC::LBoundOp>(mlirloc, builder.getIndexType(), opd1mem, dim)
            .getResult();
    auto ub =
        builder
            .create<FC::UBoundOp>(mlirloc, builder.getIndexType(), opd1mem, dim)
            .getResult();
    auto doop = builder.create<FC::DoOp>(mlirloc, attr, lb, ub, step);
    builder = doop.getBodyBuilder();
    indices.push_back(FC::SubscriptRange(doop.getIndVar()));
  }

  // Load the operands.
  auto opd1val = builder.create<FC::FCLoadOp>(mlirloc, opd1mem, indices);
  auto opd2val = builder.create<FC::FCLoadOp>(mlirloc, opd2mem, indices);
  // Do the operation
  auto sumval = builder.create<BinOp>(mlirloc, opd1val, opd2val);
  // Store result.
  builder.create<FC::FCStoreOp>(mlirloc, sumval, resultMem, indices);

  if (createNewAlloc) {
    // The result val must be re-loaded and uses replaced.
    auto resultVal = rewriter.create<FC::FCLoadOp>(mlirloc, resultMem);
    arrayop.replaceAllUsesWith(resultVal.getOperation());
  }
  rewriter.eraseOp(arrayop);

  return this->matchSuccess();
}

struct ArrayOpsLoweringPass : public FunctionPass<ArrayOpsLoweringPass> {
  virtual void runOnFunction() {
    auto M = getFunction();

    OwningRewritePatternList patterns;

    // Integer ops
    patterns.insert<ArrayBinOpLoweringPattern<FC::ArrayAddIOp, mlir::AddIOp>>(
        &getContext());
    patterns.insert<ArrayBinOpLoweringPattern<FC::ArrayMulIOp, mlir::MulIOp>>(
        &getContext());
    patterns.insert<ArrayBinOpLoweringPattern<FC::ArraySubIOp, mlir::SubIOp>>(
        &getContext());
    patterns
        .insert<ArrayBinOpLoweringPattern<FC::ArrayDivIOp, mlir::SignedDivIOp>>(
            &getContext());
    patterns
        .insert<ArrayBinOpLoweringPattern<FC::ArrayModIOp, mlir::SignedRemIOp>>(
            &getContext());
    // FP ops
    patterns.insert<ArrayBinOpLoweringPattern<FC::ArrayAddFOp, mlir::AddFOp>>(
        &getContext());
    patterns.insert<ArrayBinOpLoweringPattern<FC::ArrayMulFOp, mlir::MulFOp>>(
        &getContext());
    patterns.insert<ArrayBinOpLoweringPattern<FC::ArraySubFOp, mlir::SubFOp>>(
        &getContext());
    patterns.insert<ArrayBinOpLoweringPattern<FC::ArrayDivFOp, mlir::DivFOp>>(
        &getContext());
    patterns.insert<ArrayBinOpLoweringPattern<FC::ArrayModFOp, mlir::RemFOp>>(
        &getContext());

    applyPatternsGreedily(M, patterns);
  }
};

} // namespace mlir

/// Create a LoopTransform pass.
std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createArrayOpsLoweringPass() {
  return std::make_unique<mlir::ArrayOpsLoweringPass>();
}

static mlir::PassRegistration<mlir::ArrayOpsLoweringPass>
    pass("lower-array-ops", "Pass to convert Array op to fc.do loop");
