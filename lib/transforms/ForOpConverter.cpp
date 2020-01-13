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
//===- ForOpConverter.cpp - Loop.for to affine.for conversion -------------===//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert loop.for to affine.for
//
//===----------------------------------------------------------------------===//

#include "transforms/ForOpConverter.h"
#include "common/Debug.h"
#include "dialect/FCOps/FCOps.h"

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

#define PASS_NAME "ForOpConvert"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;
using namespace fcmlir;

// TODO: convert to OpRewritePattern form
static mlir::Value getAffineOpFor(PatternRewriter &rewriter,
                                  mlir::Value operand) {
  if (isValidDim(operand)) {
    return operand;
  }
  auto Op = operand.getDefiningOp();
  assert(Op);
  rewriter.setInsertionPoint(Op);
  if (auto constant = dyn_cast<ConstantIntOp>(Op)) {
    auto indexConstant = rewriter.create<mlir::ConstantIndexOp>(
        constant.getLoc(), constant.getValue());
    return indexConstant;
  }

  if (!isa<mlir::AddIOp>(Op) && !isa<mlir::SubIOp>(Op)) {
    FC_DEBUG(llvm::errs() << "\ngetAffineOpFor : failed for: " << *Op);
    return nullptr;
  }

  llvm::SmallVector<mlir::Value, 2> newOperands;
  for (auto operand : Op->getOperands()) {
    auto newOp = getAffineOpFor(rewriter, operand);
    if (!newOp) {
      return nullptr;
    }
    if (newOp != operand)
      operand.replaceAllUsesWith(newOp);
    newOperands.push_back(newOp);
  }

  llvm::SmallVector<mlir::AffineExpr, 2> dims;
  for (unsigned I = 0; I < newOperands.size(); ++I) {
    dims.push_back(rewriter.getAffineDimExpr(I));
  }
  llvm::SmallVector<mlir::AffineExpr, 2> results;

  AffineExpr temp;
  if (auto addOp = dyn_cast<AddIOp>(Op)) {
    temp = mlir::getAffineBinaryOpExpr(AffineExprKind::Add, dims[0], dims[1]);
    results.push_back(temp);
  } else if (auto addOp = dyn_cast<SubIOp>(Op)) {
    dims[1] = mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::Mul, dims[1],
                                          rewriter.getAffineConstantExpr(-1));
    temp = mlir::getAffineBinaryOpExpr(AffineExprKind::Add, dims[0], dims[1]);
    results.push_back(temp);
  } else if (auto addOp = dyn_cast<SignedRemIOp>(Op)) {
    temp = mlir::getAffineBinaryOpExpr(AffineExprKind::Mod, dims[0], dims[1]);
    results.push_back(temp);
  }

  auto map = mlir::AffineMap::get(dims.size(), 0, results);
  auto affineOp =
      rewriter.create<mlir::AffineApplyOp>(Op->getLoc(), map, newOperands);
  llvm::errs() << "\n Rewriting Op : " << *Op;
  llvm::errs() << "\n\t With : " << *affineOp.getOperation();
  return affineOp.getResult();
}

struct LoadOpRewriter : public OpRewritePattern<mlir::LoadOp> {
  using OpRewritePattern<mlir::LoadOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(mlir::LoadOp loadOp,
                                     PatternRewriter &rewriter) const override {
    if (!loadOp.getParentOfType<mlir::loop::ForOp>()) {
      return matchFailure();
    }
    if (loadOp.getIndices().empty()) {
      return matchFailure();
    }
    rewriter.setInsertionPoint(loadOp);
    SmallVector<mlir::Value, 2> indexOperands(loadOp.getIndices());
    for (unsigned I = 0; I < indexOperands.size(); I++) {
      auto affineOp = getAffineOpFor(rewriter, indexOperands[I]);
      if (!affineOp) {
        llvm::errs() << "\nfailed for : " << indexOperands[I];
        return matchFailure();
      }
      indexOperands[I] = affineOp;
    }
    rewriter.setInsertionPoint(loadOp);
    auto affineLoad = rewriter.create<AffineLoadOp>(
        loadOp.getLoc(), loadOp.getMemRef(), indexOperands);
    assert(affineLoad);
    rewriter.replaceOp(loadOp.getOperation(), {affineLoad.getResult()});
    return matchSuccess();
  }
};

struct StoreOpRewriter : public OpRewritePattern<mlir::StoreOp> {
  using OpRewritePattern<mlir::StoreOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(mlir::StoreOp storeOp,
                                     PatternRewriter &rewriter) const override {
    if (!storeOp.getParentOfType<mlir::loop::ForOp>()) {
      return matchFailure();
    }
    if (storeOp.getIndices().empty()) {
      return matchFailure();
    }
    rewriter.setInsertionPoint(storeOp);
    SmallVector<mlir::Value, 2> indexOperands(storeOp.getIndices());
    for (unsigned I = 0; I < indexOperands.size(); I++) {
      auto affineOp = getAffineOpFor(rewriter, indexOperands[I]);
      if (!affineOp) {
        llvm::errs() << "\nfailed for : " << indexOperands[I];
        return matchFailure();
      }
      indexOperands[I] = affineOp;
    }
    rewriter.setInsertionPoint(storeOp);
    rewriter.create<AffineStoreOp>(storeOp.getLoc(), storeOp.getValueToStore(),
                                   storeOp.getMemRef(), indexOperands);
    rewriter.replaceOp(storeOp.getOperation(), llvm::None);
    return matchSuccess();
  }
};

struct IndexCastOpSimplifier : public OpRewritePattern<mlir::IndexCastOp> {
  using OpRewritePattern<mlir::IndexCastOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(mlir::IndexCastOp op,
                                     PatternRewriter &rewriter) const override {
    auto v = op.in();
    if (!v.getDefiningOp()) {
      return matchFailure();
    }
    auto constInt = llvm::dyn_cast_or_null<ConstantIntOp>(v.getDefiningOp());
    if (constInt) {
      auto indexVal =
          rewriter.create<ConstantIndexOp>(v.getLoc(), constInt.getValue());
      rewriter.replaceOp(op.getOperation(), {indexVal.getResult()}, {constInt});
      return matchSuccess();
    }

    // %1 = indexcastop  %val to i32 : index to i32
    // %2 = indexcastop %1 to index : i32 to index
    // replace it with %val.
    auto otherOp = llvm::dyn_cast_or_null<mlir::IndexCastOp>(v.getDefiningOp());
    if (otherOp) {
      auto otherV = otherOp.in();
      if (otherV.getType() == op.getType() &&
          op.getType().isa<mlir::IndexType>()) {
        rewriter.replaceOp(op.getOperation(), {otherV}, {otherOp});
        // op.getOperation()->getResult(0).replaceAllUsesWith(otherV);
        return matchSuccess();
      }
    }

    auto castToIndex = [&](mlir::Value v) {
      return rewriter.create<mlir::IndexCastOp>(v.getLoc(), v,
                                                rewriter.getIndexType());
    };

    if (!op.getType().isa<mlir::IndexType>()) {
      return matchFailure();
    }
    auto definedOp = v.getDefiningOp();
    if (definedOp->getResult(0).getType() != rewriter.getIntegerType(32))
      return matchFailure();

    rewriter.setInsertionPoint(definedOp);
    if (auto addOp = dyn_cast<mlir::AddIOp>(definedOp)) {
      auto lhs = definedOp->getOperand(0);
      auto rhs = definedOp->getOperand(1);
      auto newLhs = castToIndex(lhs);
      auto newRhs = castToIndex(rhs);
      auto newAdd =
          rewriter.create<mlir::AddIOp>(addOp.getLoc(), newLhs, newRhs);
      rewriter.replaceOp(op.getOperation(), {newAdd}, {addOp});
      return matchSuccess();
    }

    if (auto subOp = dyn_cast<mlir::SubIOp>(definedOp)) {
      auto lhs = definedOp->getOperand(0);
      auto rhs = definedOp->getOperand(1);
      auto newLhs = castToIndex(lhs);
      auto newRhs = castToIndex(rhs);
      auto newSub =
          rewriter.create<mlir::SubIOp>(subOp.getLoc(), newLhs, newRhs);
      rewriter.replaceOp(op.getOperation(), {newSub}, {subOp});
      return matchSuccess();
    }
    return matchFailure();
  }
};

struct ForOpToAffineConverter : public OpRewritePattern<mlir::loop::ForOp> {
private:
  bool LegalityChecks(PatternRewriter &rewriter,
                      mlir::loop::ForOp forOp) const {

    auto initVal = forOp.lowerBound();
    auto endVal = forOp.upperBound();
    auto step = forOp.step();

    auto body = forOp.getBody();
    auto newInitVal = getAffineOpFor(rewriter, initVal);
    auto newEndVal = getAffineOpFor(rewriter, forOp.upperBound());
    auto newStep = getAffineOpFor(rewriter, forOp.step());

    if (newInitVal && initVal != newInitVal)
      initVal.replaceAllUsesWith(newInitVal);
    if (newEndVal && endVal != newEndVal)
      endVal.replaceAllUsesWith(newEndVal);
    if (newStep && step != newStep)
      step.replaceAllUsesWith(newStep);

    if (!newInitVal || !newEndVal || !newStep) {
      FC_DEBUG(llvm::errs()
               << "Could not find Affine bounds/Steps. skipping... \n");
      return false;
    }

    auto constOp = llvm::dyn_cast<ConstantIndexOp>(newStep.getDefiningOp());
    if (constOp && constOp.getValue() != 1) {
      FC_DEBUG(llvm::errs()
               << "\tStep value is not 1, " << constOp.getValue() << "\n");
      return false;
    }

    bool hasInvalidOps = false;
    body->walk([&](mlir::Operation *op) {
      if (isa<LoadOp>(op) || isa<StoreOp>(op) || isa<loop::IfOp>(op) ||
          isa<mlir::CallOp>(op)) {
        hasInvalidOps = true;
        llvm::errs() << "\nfailed for : " << *op;
        return;
      }
    });

    if (hasInvalidOps) {
      FC_DEBUG(llvm::errs() << "\tHas invalid Operations..skipping ...\n");
      return false;
    }

    return true;
  }

  bool PerformTransformation(PatternRewriter &rewriter,
                             loop::ForOp forOp) const {
    auto initVal = forOp.lowerBound();
    auto endVal = forOp.upperBound();

    SmallVector<mlir::Value, 2> lbs{initVal}, ubs{endVal};
    mlir::AffineMap lbMap = rewriter.getDimIdentityMap();
    mlir::AffineMap ubMap = rewriter.getDimIdentityMap();

    mlir::fullyComposeAffineMapAndOperands(&lbMap, &lbs);
    mlir::fullyComposeAffineMapAndOperands(&ubMap, &ubs);

    rewriter.setInsertionPoint(forOp);
    auto affineForOp = rewriter.create<mlir::AffineForOp>(forOp.getLoc(), lbs,
                                                          lbMap, ubs, ubMap, 1);

    rewriter.setInsertionPointToStart(affineForOp.getBody());

    auto begin = forOp.getBody()->begin();
    auto nOps = forOp.getBody()->getOperations().size();
    auto iv = forOp.getInductionVar();

    affineForOp.getBody()->getOperations().splice(
        affineForOp.getBody()->getOperations().begin(),
        forOp.getBody()->getOperations(), begin, std::next(begin, nOps - 1));

    replaceAllUsesInRegionWith(iv, affineForOp.getInductionVar(),
                               affineForOp.region());

    rewriter.replaceOp(forOp, llvm::None);
    return true;
  }

public:
  using OpRewritePattern<mlir::loop::ForOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(mlir::loop::ForOp forOp,
                                     PatternRewriter &rewriter) const override {
    FC_DEBUG(llvm::errs() << "Checking legality for \n");
    if (!LegalityChecks(rewriter, forOp)) {
      FC_DEBUG(llvm::errs()
               << "LEGALITY CHECKS failed. skipping this for op\n");
      return matchFailure();
    }
    FC_DEBUG(llvm::errs() << "Legal to convert\n");
    if (PerformTransformation(rewriter, forOp))
      return matchSuccess();
    FC_DEBUG(
        llvm::errs() << "Success, Current loop.for converted to affine.for\n");
    return matchFailure();
  }
};

void ForOpConverter::runOnFunction() {
  auto theFunction = getFunction();
  auto context = theFunction.getContext();

  OwningRewritePatternList patterns;

  patterns.insert<LoadOpRewriter, StoreOpRewriter>(context);
  applyPatternsGreedily(theFunction, patterns);

  patterns.clear();
  patterns.insert<ForOpToAffineConverter>(context);
  // Also add canonicalize patterns.
  AffineForOp::getCanonicalizationPatterns(patterns, context);
  AffineLoadOp::getCanonicalizationPatterns(patterns, context);
  AffineStoreOp::getCanonicalizationPatterns(patterns, context);
  applyPatternsGreedily(theFunction, patterns);
}

/// Create a LoopTransform pass.
std::unique_ptr<OpPassBase<FuncOp>> createForOpConverterPass() {
  return std::make_unique<ForOpConverter>();
}

static PassRegistration<ForOpConverter>
    pass("forop-convert", "Pass to convert loop.for to affine.for");
