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
//===- LowerProgramUnit.cpp - ---------------------------------------------===//
// 1. Converts High level function like  operations to std.func.
// 2. fc.call to std.call
// 3. Flattens the nested structure. All functions are now directly under
// ModuleOp
// 4. fc.fortran_module is removed and its variables are now fc.global.
// 5. Emits a main function if it sees MainProgram in the current transalation
// unit.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "common/Debug.h"
#include "dialect/FC/FCOps.h"

using namespace std;

namespace mlir {

struct SymbolNameMangler {
  static std::string mangle(SymbolRefAttr attr) {
    std::string mangledName;
    for (auto ref : llvm::reverse(attr.getNestedReferences())) {
      mangledName += ref.getValue().str() + ".";
    }
    return mangledName + attr.getRootReference().str();
  }
  static std::string mangle(mlir::Operation *op, std::string suffix = "") {
    if (llvm::isa<mlir::ModuleOp>(op)) {
      return suffix;
    }
    std::string currName;
    if (auto funcOp = llvm::dyn_cast<FC::FCFuncOp>(op)) {
      currName = funcOp.getName().str();
    } else if (auto mlirFuncOp = llvm::dyn_cast<mlir::FuncOp>(op)) {
      currName = mlirFuncOp.getName().str();
    } else if (auto modOp = llvm::dyn_cast<FC::FortranModuleOp>(op)) {
      currName = modOp.getName().str();
    } else if (auto allocOp = llvm::dyn_cast<FC::AllocaOp>(op)) {
      currName = allocOp.getName().str();
    } else {
      assert(false && "unknown symbol type");
    }
    if (!suffix.empty())
      currName = currName + "." + suffix;
    return mangle(op->getParentOp(), currName);
  }
};

// Dumps the main function. It simply calls the MainProgram.
static void dumpMain(FC::FCFuncOp MainProgFn, PatternRewriter &builder) {

  auto insertPt = builder.saveInsertionPoint();
  auto module = MainProgFn.getParentOfType<ModuleOp>();
  builder.setInsertionPointToStart(module.getBody());

  // Build argument for main function.
  auto loc = builder.getUnknownLoc();
  auto I32 = builder.getIntegerType(32);
  auto argcTy = FC::RefType::get(I32);
  auto I8 = builder.getIntegerType(8);
  FC::ArrayType::Shape shape(2);
  auto argvTy = FC::RefType::get(FC::RefType::get(I8));
  mlir::Attribute attr;
  auto argvPtrTy = FC::RefType::get(argvTy);
  auto argv = builder.create<FC::GlobalOp>(loc, argvPtrTy, false,
                                           "fc.internal.argv", attr);
  auto argc = builder.create<FC::GlobalOp>(loc, argcTy, false,
                                           "fc.internal.argc", attr);

  // Build main function.
  builder.setInsertionPointAfter(MainProgFn);
  auto funcType =
      builder.getFunctionType({I32, argvTy}, {builder.getIntegerType(32)});
  auto mlirFunc = builder.create<FC::FCFuncOp>(loc, "main", funcType);

  auto EntryBB = mlirFunc.addEntryBlock();
  builder.setInsertionPointToEnd(EntryBB);
  auto argvPtr = builder.create<FC::AddressOfOp>(loc, argv);
  auto argcPtr = builder.create<FC::AddressOfOp>(loc, argc);

  auto one = builder.create<mlir::ConstantIntOp>(loc, 1, 32);
  auto argcVal =
      builder.create<mlir::SubIOp>(loc, mlirFunc.getArgument(0), one);
  builder.create<FC::FCStoreOp>(loc, argcVal, argcPtr);
  builder.create<FC::FCStoreOp>(loc, mlirFunc.getArgument(1), argvPtr);

  auto Call = builder.create<FC::FCCallOp>(loc, MainProgFn);
  SmallVector<mlir::Value, 2> ops = {Call.getResult(0)};
  builder.create<FC::FCReturnOp>(loc, ops);

  builder.restoreInsertionPoint(insertPt);
}

struct ProgramUnitLoweringPattern : public OpRewritePattern<FC::FCFuncOp> {
  using OpRewritePattern<FC::FCFuncOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(FC::FCFuncOp op,
                                     PatternRewriter &rewriter) const override {
    auto funcType = op.getType();
    auto name = SymbolNameMangler::mangle(op);
    llvm::SmallVector<Type, 2> inputs;

    // Dump main function if it is a Fortran Program kind.
    auto attr = op.getAttr("main_program");
    if (attr && attr.cast<BoolAttr>().getValue()) {
      dumpMain(op, rewriter);
    }

    inputs.append(funcType.getInputs().begin(), funcType.getInputs().end());
    op.setName("old");
    auto newFuncType = rewriter.getFunctionType(inputs, funcType.getResults());
    auto mlirFunc = mlir::FuncOp::create(op.getLoc(), name, newFuncType);
    auto module = op.getParentOfType<ModuleOp>();
    auto &region = op.body();
    mlirFunc.getBody().takeBody(region);
    rewriter.eraseOp(op);
    module.push_back(mlirFunc);
    return matchSuccess();
  }
};

struct StaticAllocaOpLowering : public OpRewritePattern<FC::AllocaOp> {
  using OpRewritePattern<FC::AllocaOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(FC::AllocaOp op,
                                     PatternRewriter &rewriter) const override {
    auto type = op.getType();
    auto attr = op.getAttr("alloc_kind");
    if (!attr || !attr.isa<StringAttr>()) {
      return matchSuccess();
    }
    if (attr.cast<StringAttr>().getValue() != "static")
      return matchSuccess();
    auto name = SymbolNameMangler::mangle(op);
    auto module = op.getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToStart(module.getBody());
    mlir::Attribute value = op.getAttr("value");
    auto globalOp =
        rewriter.create<FC::GlobalOp>(op.getLoc(), type, false, name, value);
    rewriter.setInsertionPointAfter(op);

    // Address of op is not required inside fortran module op.
    if (llvm::isa<FC::FortranModuleOp>(op.getParentOp())) {
      rewriter.eraseOp(op);
      return matchSuccess();
    }
    rewriter.replaceOpWithNewOp<FC::AddressOfOp>(op, globalOp);
    return matchSuccess();
  }
};

struct FCCallOpLowering : public OpRewritePattern<FC::FCCallOp> {
  using OpRewritePattern<FC::FCCallOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(FC::FCCallOp op,
                                     PatternRewriter &rewriter) const override {
    auto symref = op.getCallee();
    auto callee = SymbolNameMangler::mangle(symref);
    llvm::SmallVector<Type, 2> results(op.getResultTypes());
    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, callee, results, llvm::SmallVector<Value, 2>(op.operands()));
    return matchSuccess();
  }
};

struct FCReturnOpLowering : public OpRewritePattern<FC::FCReturnOp> {
  using OpRewritePattern<FC::FCReturnOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(FC::FCReturnOp op,
                                     PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(
        op, llvm::SmallVector<Value, 2>(op.operands()));
    return matchSuccess();
  }
};

struct FortranModuleOpLowering : public OpRewritePattern<FC::FortranModuleOp> {
  using OpRewritePattern<FC::FortranModuleOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(FC::FortranModuleOp op,
                                     PatternRewriter &rewriter) const override {
    if (op.body().front().getOperations().size() != 1) {
      return matchFailure();
    }
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct CaptureArgLowering : public OpRewritePattern<FC::CaptureArgOp> {
  using OpRewritePattern<FC::CaptureArgOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(FC::CaptureArgOp op,
                                     PatternRewriter &rewriter) const override {
    op.replaceAllUsesWith(op.operand());
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct AddressOfOpLowering : public OpRewritePattern<FC::GetElementRefOp> {
  using OpRewritePattern<FC::GetElementRefOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(FC::GetElementRefOp op,
                                     PatternRewriter &rewriter) const override {
    auto symref = op.getSymRef();
    auto module = op.getParentOfType<mlir::ModuleOp>();
    auto varName = SymbolNameMangler::mangle(symref);
    auto global = module.lookupSymbol<FC::GlobalOp>(varName);
    if (!global) {
      return matchFailure();
    }
    rewriter.replaceOpWithNewOp<FC::AddressOfOp>(op, global);
    return matchSuccess();
  }
}; // namespace mlir

struct ProgramUnitLowering : public ModulePass<ProgramUnitLowering> {
  virtual void runOnModule() {

    OwningRewritePatternList patterns;
    patterns.insert<ProgramUnitLoweringPattern>(&getContext());
    patterns.insert<FortranModuleOpLowering>(&getContext());
    patterns.insert<FCCallOpLowering>(&getContext());
    patterns.insert<FCReturnOpLowering>(&getContext());
    patterns.insert<StaticAllocaOpLowering>(&getContext());
    patterns.insert<AddressOfOpLowering>(&getContext());
    patterns.insert<CaptureArgLowering>(&getContext());
    applyPatternsGreedily(getModule(), patterns);
    return;
  }
};
} // namespace mlir

std::unique_ptr<mlir::Pass> createProgramUnitLoweringPass() {
  return std::make_unique<mlir::ProgramUnitLowering>();
}

static mlir::PassRegistration<mlir::ProgramUnitLowering>
    pass("lower-pu", "Pass to lower program unit");
