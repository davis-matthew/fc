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
//===- ForOpConverter.cpp - fc.do to affine.for conversion -------------===//
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert fc.do to affine.for
//
//===----------------------------------------------------------------------===//

#include "transforms/FCDoConverter.h"
#include "common/Debug.h"

#include "dialect/FC/FCOps.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

#define PASS_NAME "FcDoConvert"
#define DEBUG_TYPE PASS_NAME
// TODO: Add DEBUG WITH TYPE and STATISTIC

using namespace fc;
using namespace llvm;
using namespace fcmlir;

static mlir::Value castToIndexTy(mlir::Location mlirloc,
                                 PatternRewriter &rewriter, mlir::Value value) {
  auto indexTy = rewriter.getIndexType();
  if (value.getType() == indexTy)
    return value;

  rewriter.setInsertionPointAfter(value.getDefiningOp());
  return rewriter.create<IndexCastOp>(mlirloc, value, indexTy);
}

static mlir::Value getAffineAdd(mlir::Location mlirloc, mlir::Value op1,
                                mlir::Value op2, PatternRewriter &rewriter) {

  op1 = castToIndexTy(mlirloc, rewriter, op1);
  op2 = castToIndexTy(mlirloc, rewriter, op2);
  llvm::SmallVector<mlir::AffineExpr, 2> dims;
  for (unsigned I = 0; I < 2; ++I) {
    dims.push_back(rewriter.getAffineDimExpr(I));
  }

  auto addExpr =
      mlir::getAffineBinaryOpExpr(AffineExprKind::Add, dims[0], dims[1]);
  llvm::SmallVector<mlir::Value, 2> operands{op1, op2};

  llvm::SmallVector<mlir::AffineExpr, 2> results{addExpr};
  auto map = mlir::AffineMap::get(dims.size(), 0, results);
  auto affineOp = rewriter.create<mlir::AffineApplyOp>(mlirloc, map, operands);
  return affineOp;
}

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
    mlir::Value indexConstant = rewriter.create<mlir::ConstantIndexOp>(
        constant.getLoc(), constant.getValue());
    return indexConstant;
  }

  if (auto fcLoad = dyn_cast<FC::FCLoadOp>(Op)) {
    if (!(fcLoad.getParentOfType<FC::DoOp>() ||
          fcLoad.getParentOfType<AffineForOp>())) {
      return castToIndexTy(Op->getLoc(), rewriter, fcLoad);
    }
    mlir::Value pointer = fcLoad.getPointer();
    rewriter.setInsertionPoint(Op);
    auto castOp = rewriter.create<FC::CastToMemRefOp>(fcLoad.getLoc(), pointer);

    SmallVector<mlir::Value, 2> indexOperands(fcLoad.getIndices());
    for (unsigned I = 0; I < indexOperands.size(); I++) {
      if (!isValidDim(indexOperands[I])) {
        return nullptr;
      }
    }

    auto affineLoad = rewriter.create<AffineLoadOp>(
        fcLoad.getLoc(), castOp.getResult(), indexOperands);
    return affineLoad;
  }

  if (!isa<mlir::AddIOp>(Op) && !isa<mlir::SubIOp>(Op)) {
    LLVM_DEBUG(llvm::errs() << "\ngetAffineOpFor : failed for: " << *Op);
    return nullptr;
  }

  llvm::SmallVector<mlir::Value, 2> newOperands;
  for (auto operand : Op->getOperands()) {
    auto newOp = getAffineOpFor(rewriter, operand);
    if (!newOp) {
      return nullptr;
    }
    newOp = castToIndexTy(Op->getLoc(), rewriter, newOp);
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
  } else if (auto addOp = dyn_cast<mlir::SignedRemIOp>(Op)) {
    temp = mlir::getAffineBinaryOpExpr(AffineExprKind::Mod, dims[0], dims[1]);
    results.push_back(temp);
  }

  auto map = mlir::AffineMap::get(dims.size(), 0, results);
  auto affineOp =
      rewriter.create<mlir::AffineApplyOp>(Op->getLoc(), map, newOperands);
  LLVM_DEBUG(llvm::errs() << "\n Rewriting Op : " << *Op;
             llvm::errs() << "\n\t With : " << *affineOp.getOperation(););
  return affineOp.getResult();
}

struct FCLoadToAffineConverter : public OpRewritePattern<FC::FCLoadOp> {
  using OpRewritePattern<FC::FCLoadOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(FC::FCLoadOp loadOp,
                                     PatternRewriter &rewriter) const override {

    if (!(loadOp.getParentOfType<FC::DoOp>() ||
          loadOp.getParentOfType<AffineForOp>())) {
      return matchFailure();
    }

    if (loadOp.getIndices().empty())
      return matchFailure();

    mlir::Value pointer = loadOp.getPointer();
    rewriter.setInsertionPoint(loadOp);
    auto castOp = rewriter.create<FC::CastToMemRefOp>(loadOp.getLoc(), pointer);

    SmallVector<mlir::Value, 2> indexOperands(loadOp.getIndices());
    for (unsigned I = 0; I < indexOperands.size(); I++) {
      if (!isValidDim(indexOperands[I])) {
        return matchFailure();
      }
    }

    rewriter.setInsertionPoint(loadOp);
    auto affineLoad = rewriter.create<AffineLoadOp>(
        loadOp.getLoc(), castOp.getResult(), indexOperands);
    assert(affineLoad);
    rewriter.replaceOp(loadOp, {affineLoad.getResult()});
    return matchSuccess();
  }
};

struct FCStoreToAffineConverter : public OpRewritePattern<FC::FCStoreOp> {
  using OpRewritePattern<FC::FCStoreOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(FC::FCStoreOp storeOp,
                                     PatternRewriter &rewriter) const override {

    LLVM_DEBUG(llvm::errs() << "Trying to rewrite " << *storeOp << "\n");
    if (!(storeOp.getParentOfType<FC::DoOp>() ||
          storeOp.getParentOfType<AffineForOp>())) {
      llvm::errs().indent(2) << "Parent is not for\n";
      return matchFailure();
    }

    if (storeOp.getIndices().empty())
      return matchFailure();

    mlir::Value pointer = storeOp.getPointer();
    auto castOp =
        rewriter.create<FC::CastToMemRefOp>(storeOp.getLoc(), pointer);

    rewriter.setInsertionPoint(storeOp);
    SmallVector<mlir::Value, 2> indexOperands(storeOp.getIndices());
    for (unsigned I = 0; I < indexOperands.size(); I++) {
      if (!isValidDim(indexOperands[I])) {
        LLVM_DEBUG(llvm::errs().indent(2) << "Operands in not valid dim " << I
                                          << " " << indexOperands[I] << "\n");
        return matchFailure();
      }
    }

    rewriter.setInsertionPoint(storeOp);
    rewriter.create<AffineStoreOp>(storeOp.getLoc(), storeOp.getValueToStore(),
                                   castOp.getResult(), indexOperands);
    rewriter.replaceOp(storeOp.getOperation(), llvm::None);
    return matchSuccess();
  }
};

template <class ExprTy, AffineExprKind affineExprKind>
struct BinaryExprTranslator : public OpRewritePattern<ExprTy> {
  using OpRewritePattern<ExprTy>::OpRewritePattern;

  void makeConstantIdxIfPossible(PatternRewriter &rewriter,
                                 mlir::Value &val) const {
    if (isValidDim(val)) {
      return;
    }

    // This can happen in case of block-args
    if (!val.getDefiningOp()) {
      val = val.getType().isa<IndexType>() ? val : nullptr;
      return;
    }

    if (auto constant = dyn_cast<ConstantIntOp>(val.getDefiningOp())) {
      auto indexConstant = rewriter.create<mlir::ConstantIndexOp>(
          constant.getLoc(), constant.getValue());
      val = indexConstant;
      return;
    }
    val = nullptr;
  }

  PatternMatchResult matchAndRewrite(ExprTy exprOp,
                                     PatternRewriter &rewriter) const override {
    mlir::Value lhs = exprOp.lhs();
    mlir::Value rhs = exprOp.rhs();
    makeConstantIdxIfPossible(rewriter, lhs);
    makeConstantIdxIfPossible(rewriter, rhs);

    // FIXME: the this-> is weird!, this is due to the template arg to the
    // template parent-class. Not sure how to fix this.
    if (!lhs || !rhs) {
      return this->matchFailure();
    }

    llvm::SmallVector<mlir::AffineExpr, 2> results;
    llvm::SmallVector<mlir::AffineExpr, 2> dims;
    dims.push_back(rewriter.getAffineDimExpr(0));
    dims.push_back(rewriter.getAffineDimExpr(1));

    // There's no AffineExprKind::Sub :/
    if (llvm::isa<SubIOp>(exprOp.getOperation())) {
      dims[1] = mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::Mul, dims[1],
                                            rewriter.getAffineConstantExpr(-1));
    }

    AffineExpr affineExpr =
        getAffineBinaryOpExpr(affineExprKind, dims[0], dims[1]);

    results.push_back(affineExpr);

    llvm::SmallVector<mlir::Value, 2> operands{lhs, rhs};
    auto map = mlir::AffineMap::get(dims.size(), 0, results);
    auto affineOp =
        rewriter.create<mlir::AffineApplyOp>(exprOp.getLoc(), map, operands);
    rewriter.replaceOp(exprOp.getOperation(), {affineOp.getResult()});
    return this->matchSuccess();
  }
};

struct FCDoToAffineConverter : public OpRewritePattern<FC::DoOp> {
private:
  bool LegalityChecks(PatternRewriter &rewriter, FC::DoOp fcDo) const {
    auto initVal = fcDo.getLowerBound();
    auto endVal = fcDo.getUpperBound();
    auto step = fcDo.getStep();

    auto body = fcDo.getBody();

    bool hasInvalidOps = false;
    body->walk([&](mlir::Operation *op) {
      if (isa<LoadOp>(op) || isa<StoreOp>(op) || isa<loop::IfOp>(op) ||
          isa<mlir::CallOp>(op)) {
        hasInvalidOps = true;
        LLVM_DEBUG(llvm::errs() << "\nfailed for : " << *op;);
        return;
      }
    });

    if (hasInvalidOps) {
      LLVM_DEBUG(llvm::errs() << "\tHas invalid Operations..skipping ...\n");
      return false;
    }

    /// FIXME: This FCLoadOp and FCStoreOp walker can be merged with above
    //         general operation walker. But it seems to increase compilation
    //         time  heavily!
    body->walk([&](FC::FCLoadOp) {
      hasInvalidOps = true;
      LLVM_DEBUG(llvm::errs() << "FC load op found. Skipping... \n");
      return;
    });

    if (!hasInvalidOps) {
      body->walk([&](FC::FCStoreOp) {
        hasInvalidOps = true;
        LLVM_DEBUG(llvm::errs() << "FC store op found. Skipping... \n");
        return;
      });
    }

    if (hasInvalidOps) {
      LLVM_DEBUG(llvm::errs() << "\tHas invalid Operations..skipping ...\n");
      return false;
    }

    auto newInitVal = getAffineOpFor(rewriter, initVal);
    auto newEndVal = getAffineOpFor(rewriter, fcDo.getUpperBound());
    auto newStep = getAffineOpFor(rewriter, fcDo.getStep());

    if (!newInitVal || !newEndVal || !newStep) {
      LLVM_DEBUG(llvm::errs()
                 << "Could not find Affine bounds/Steps. skipping... \n");
      return false;
    }

    auto constOp = llvm::dyn_cast<ConstantIndexOp>(newStep.getDefiningOp());
    if (constOp && constOp.getValue() != 1) {
      LLVM_DEBUG(llvm::errs()
                 << "\tStep value is not 1, " << constOp.getValue() << "\n");
      return false;
    }

    if (newInitVal && initVal != newInitVal)
      fcDo.setLowerBound(newInitVal);
    if (newEndVal && endVal != newEndVal)
      fcDo.setUpperBound(newEndVal);
    if (newStep && step != newStep)
      fcDo.setStep(newStep);
    return true;
  }

  bool PerformTransformation(PatternRewriter &rewriter, FC::DoOp fcDo) const {
    auto initVal = fcDo.getLowerBound();
    mlir::Value endVal = fcDo.getUpperBound();
    mlir::Value step = fcDo.getStep();

    rewriter.setInsertionPoint(fcDo);
    auto mlirloc = fcDo.getLoc();
    endVal = getAffineAdd(mlirloc, step, endVal, rewriter);

    SmallVector<mlir::Value, 2> lbs{initVal}, ubs{endVal};
    mlir::AffineMap lbMap = rewriter.getDimIdentityMap();
    mlir::AffineMap ubMap = rewriter.getDimIdentityMap();

    mlir::fullyComposeAffineMapAndOperands(&lbMap, &lbs);
    mlir::fullyComposeAffineMapAndOperands(&ubMap, &ubs);

    rewriter.setInsertionPoint(fcDo);
    auto affineForOp = rewriter.create<mlir::AffineForOp>(fcDo.getLoc(), lbs,
                                                          lbMap, ubs, ubMap, 1);
    rewriter.setInsertionPointToStart(affineForOp.getBody());

    auto begin = fcDo.getBody()->begin();
    auto nOps = fcDo.getBody()->getOperations().size();
    auto iv = fcDo.getIndVar();

    affineForOp.getBody()->getOperations().splice(
        affineForOp.getBody()->getOperations().begin(),
        fcDo.getBody()->getOperations(), begin, std::next(begin, nOps - 1));
    replaceAllUsesInRegionWith(iv, affineForOp.getInductionVar(),
                               affineForOp.region());
    rewriter.replaceOp(fcDo, llvm::None);
    llvm::errs() << "FC::Do  onverted\n";
    return true;
  }

public:
  using OpRewritePattern<FC::DoOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(FC::DoOp fcDo,
                                     PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::errs() << "Checking legality for \n");
    if (!LegalityChecks(rewriter, fcDo)) {
      LLVM_DEBUG(llvm::errs()
                 << "LEGALITY CHECKS failed. skipping this fc do\n");
      return matchFailure();
    }
    LLVM_DEBUG(llvm::errs() << "Legal to convert\n");
    if (PerformTransformation(rewriter, fcDo))
      return matchSuccess();
    LLVM_DEBUG(llvm::errs()
               << "Success, Current fc.do converted to affine.for\n");
    return matchFailure();
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

void FCDoConverter::runOnFunction() {
  auto theFunction = getFunction();
  auto context = theFunction.getContext();

  OwningRewritePatternList patterns;
  patterns.insert<IndexCastOpSimplifier, FCLoadToAffineConverter,
                  FCStoreToAffineConverter>(context);
  patterns.insert<BinaryExprTranslator<AddIOp, AffineExprKind::Add>>(context);
  patterns.insert<BinaryExprTranslator<SubIOp, AffineExprKind::Add>>(context);
  patterns.insert<BinaryExprTranslator<MulIOp, AffineExprKind::Mul>>(context);
  patterns
      .insert<BinaryExprTranslator<mlir::SignedRemIOp, AffineExprKind::Mod>>(
          context);

  applyPatternsGreedily(theFunction, patterns);

  patterns.clear();

  patterns.insert<FCDoToAffineConverter>(context);
  // Also add canonicalize patterns.
  AffineForOp::getCanonicalizationPatterns(patterns, context);
  AffineLoadOp::getCanonicalizationPatterns(patterns, context);
  AffineStoreOp::getCanonicalizationPatterns(patterns, context);
  applyPatternsGreedily(theFunction, patterns);
}

std::unique_ptr<OpPassBase<FuncOp>> createFCDoConverterPass() {
  return std::make_unique<FCDoConverter>();
}

static PassRegistration<FCDoConverter>
    pass("fcdo-convert", "Pass to convert fc.do to affine.for");
