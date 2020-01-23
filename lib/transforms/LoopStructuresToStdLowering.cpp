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
//===- LoopStructureLoweringPass.cpp - convert Loops to CFG  --------------===//
//
//===----------------------------------------------------------------------===//
//
// Lowers all loop like operations to CFG based loops.
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

struct DOLoopInfo {
  mlir::Block *latch{nullptr};
  mlir::Block *exit{nullptr};
};

struct DoOpLoweringPattern : public OpRewritePattern<FC::DoOp> {
public:
  using OpRewritePattern<FC::DoOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(FC::DoOp doOp,
                                     PatternRewriter &rewriter) const override;
}; // namespace mlir

static void collectControlFlowOpsToReplace(
    FC::DoOp doOp, llvm::SmallVectorImpl<mlir::Operation *> &opsToReplace) {
  doOp.walk([&](Operation *op) {
    llvm::StringRef name;
    auto pdoOp = op->getParentOfType<FC::DoOp>();
    auto directlyInsideThisLoop = pdoOp && (pdoOp == doOp);
    if (auto cycleOp = llvm::dyn_cast<FC::CycleOp>(op)) {
      name = cycleOp.getConstructName();
    } else if (auto exitOp = llvm::dyn_cast<FC::ExitOp>(op)) {
      name = exitOp.getConstructName();
    } else if (auto retOp = llvm::dyn_cast<FC::DoReturnOp>(op)) {
      if (directlyInsideThisLoop)
        opsToReplace.push_back(retOp);
      return;
    } else {
      return;
    }

    // If it belongs to current loop we are in.
    auto hasToReplace = (name.empty() && directlyInsideThisLoop);
    // OR If the name is current loop we are in.
    hasToReplace |= (!name.empty() && name.equals(doOp.getConstructName()));
    if (hasToReplace) {
      opsToReplace.push_back(op);
    }
  });
}

static void
replaceControlFlowOps(llvm::SmallVectorImpl<Operation *> &opsToReplace,
                      PatternRewriter &rewriter, DOLoopInfo &loopBlockInfo) {
  for (auto *op : opsToReplace) {
    rewriter.setInsertionPointAfter(op);
    if (auto retOp = llvm::dyn_cast<FC::DoReturnOp>(op)) {
      rewriter.create<mlir::ReturnOp>(
          op->getLoc(), SmallVector<mlir::Value, 2>(retOp.operand_begin(),
                                                    retOp.operand_end()));
    } else {
      mlir::Block *toBlock = loopBlockInfo.latch;
      if (llvm::isa<FC::ExitOp>(op)) {
        toBlock = loopBlockInfo.exit;
      }
      rewriter.create<BranchOp>(op->getLoc(), toBlock, llvm::None);
    }
    // create dead block.
    rewriter.splitBlock(rewriter.getInsertionBlock(),
                        rewriter.getInsertionPoint());
    rewriter.eraseOp(op);
  }
} // namespace mlir

PatternMatchResult
DoOpLoweringPattern::matchAndRewrite(FC::DoOp doOp,
                                     PatternRewriter &rewriter) const {

  auto loc = doOp.getLoc();

  llvm::SmallVector<mlir::Operation *, 2> opsToReplace;
  collectControlFlowOpsToReplace(doOp, opsToReplace);

  auto castToIndex = [&](mlir::Value val) {
    if (!val) {
      return val;
    }
    if (val.getType().isIndex())
      return val;
    return rewriter.create<mlir::IndexCastOp>(loc, val, rewriter.getIndexType())
        .getResult();
  };
  auto lb = castToIndex(doOp.getLowerBound());
  auto ub = castToIndex(doOp.getUpperBound());
  auto step = castToIndex(doOp.getStep());
  auto hasExpr = lb && ub && step;
  if (hasExpr && (!lb || !ub || !step))
    return matchFailure();

  // prepare LLVM like loop structure.
  auto *preheader = rewriter.getInsertionBlock();
  auto initPosition = rewriter.getInsertionPoint();
  auto *exitBlock = rewriter.splitBlock(preheader, initPosition);
  auto *header = &doOp.region().front();
  auto *firstBodyBlock = rewriter.splitBlock(header, header->begin());
  auto *lastBlock = &doOp.region().back();
  auto *latch =
      rewriter.splitBlock(lastBlock, lastBlock->getTerminator()->getIterator());
  rewriter.setInsertionPointToEnd(lastBlock);
  rewriter.create<BranchOp>(loc, latch);

  rewriter.inlineRegionBefore(doOp.region(), exitBlock);
  auto iv = header->getArgument(0);

  DOLoopInfo loopBlockInfo{latch, exitBlock};
  replaceControlFlowOps(opsToReplace, rewriter, loopBlockInfo);

  if (!hasExpr) {
    header->eraseArgument(0);

    // Build preheader.
    rewriter.setInsertionPointToEnd(preheader);
    rewriter.create<BranchOp>(loc, header);

    rewriter.setInsertionPointToEnd(header);
    rewriter.create<BranchOp>(loc, firstBodyBlock);

    rewriter.setInsertionPointToEnd(latch);
    rewriter.create<BranchOp>(loc, header);

    rewriter.eraseOp(doOp);
    return matchSuccess();
  }

  // New Block argument for trip count induction variable.
  auto tripCountIV = header->addArgument(rewriter.getIndexType());

  // Build preheader.
  rewriter.setInsertionPointToEnd(preheader);
  // trip count = (ub -lb + s)/ s
  auto tripCount = rewriter.create<mlir::SubIOp>(loc, ub, lb).getResult();
  tripCount = rewriter.create<mlir::AddIOp>(loc, tripCount, step).getResult();
  tripCount =
      rewriter.create<mlir::SignedDivIOp>(loc, tripCount, step).getResult();
  rewriter.create<BranchOp>(loc, header,
                            ArrayRef<mlir::Value>({lb, tripCount}));

  // Build Latch condition.
  rewriter.setInsertionPointToEnd(latch);
  auto stepValue = rewriter.create<AddIOp>(loc, iv, step).getResult();
  auto one = rewriter.create<mlir::ConstantIndexOp>(loc, 1);
  auto tripCountVal =
      rewriter.create<SubIOp>(loc, tripCountIV, one).getResult();
  rewriter.create<BranchOp>(loc, header,
                            ArrayRef<mlir::Value>({stepValue, tripCountVal}));

  // Build header.
  rewriter.setInsertionPointToEnd(header);
  auto zero = rewriter.create<mlir::ConstantIndexOp>(loc, 0);
  auto comparison =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt, tripCountIV, zero);
  rewriter.create<CondBranchOp>(loc, comparison, firstBodyBlock, llvm::None,
                                exitBlock, llvm::None);

  rewriter.eraseOp(doOp);
  return matchSuccess();
}

struct LoopStructureLoweringPass
    : public OperationPass<LoopStructureLoweringPass, mlir::FuncOp> {
  virtual void runOnOperation() {
    auto M = getOperation();

    OwningRewritePatternList patterns;

    patterns.insert<DoOpLoweringPattern>(&getContext());
    populateAffineToStdConversionPatterns(patterns, &getContext());
    populateLoopToStdConversionPatterns(patterns, &getContext());

    applyPatternsGreedily(M, patterns);
  }
};
} // namespace mlir

/// Create a LoopTransform pass.
std::unique_ptr<Pass> createLoopStructureLoweringPass() {
  return std::make_unique<mlir::LoopStructureLoweringPass>();
}

static mlir::PassRegistration<mlir::LoopStructureLoweringPass>
    pass("lower-loop-ops", "Pass to convert fc.do CFG based loop");
