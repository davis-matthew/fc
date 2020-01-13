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
//===- SimplifyLoopMemOps.cpp ---------------------------------------------===//
// 1. Hoist loads out of loop nest.
// 2. Perform mem2reg for some specific patterns inside loop
// TODO: Generalize.
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Dominance.h"
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
#include "dialect/FCOps/FCOps.h"
#include "transforms/AliasAnalysis.h"

#define DEBUG_TYPE "simplify-loop-mem"

using namespace std;

namespace mlir {

struct SimplifyLoopMemOps
    : public OperationPass<SimplifyLoopMemOps, FC::FCFuncOp> {

  void hoistInvariantLoads(FC::DoOp op) {
    fcmlir::AliasSetTracker AST;
    AST.add(op.region());

    llvm::SmallVector<mlir::Operation *, 2> opsToHoist;
    op.walk([&](FC::FCLoadOp loadOp) {
      auto pointer = loadOp.getPointer();
      // Cannot hoist global load.
      // TODO: Do simple analysis like,there are no function
      // calls, etc.
      if (llvm::isa_and_nonnull<FC::AddressOfOp>(pointer.getDefiningOp())) {
        return;
      }
      auto AS = AST.getAliasSetFor(loadOp);
      if (AS && AS->isMod()) {
        return;
      }
      // Check if the operands of load are invariants.
      for (auto operand : loadOp.getOperands()) {
        if (!op.isDefinedOutsideOfLoop(operand)) {
          return;
        }
      }

      opsToHoist.push_back(loadOp);
      LLVM_DEBUG(llvm::errs()
                 << "\n Hoisting load : " << *loadOp.getOperation());
    });

    op.walk([&](FC::FCStoreOp storeOp) {
      auto pointer = storeOp.getPointer();
      // Cannot hoist global load.
      // TODO: Do simple analysis like,there are no function
      // calls, etc.
      if (llvm::isa_and_nonnull<FC::AddressOfOp>(pointer.getDefiningOp())) {
        return;
      }

      // Check if the operands of load are invariants.
      for (auto operand : storeOp.getOperands()) {
        if (!op.isDefinedOutsideOfLoop(operand)) {
          return;
        }
      }

      for (auto user : pointer.getUsers()) {
        if (user == storeOp)
          continue;
        for (unsigned i = 0; i < user->getNumResults(); ++i)
          if (!op.isDefinedOutsideOfLoop(user->getResult(i))) {
            return;
          }
      }

      opsToHoist.push_back(storeOp);
    });

    op.moveOutOfLoop(opsToHoist);
  }

  // Find the store which dominates loads in the loop nest op.
  bool canPromoteMemToReg(FC::FCStoreOp storeOp, FC::DoOp op,
                          fcmlir::AliasSetTracker &AST,
                          mlir::DominanceInfo &DT) {

    LLVM_DEBUG(llvm::errs()
               << "checking safety to promote " << *storeOp << "\n");
    // scalar memrefs only.
    if (!storeOp.getIndices().empty()) {
      LLVM_DEBUG(llvm::errs().indent(2) << "Array store\n");
      return false;
    }

    // Get the alias set for the store op in loop.
    auto AS = AST.getAliasSetFor(storeOp);

    // if no alias or confused alias, return false;
    if (AS->resultKind == fcmlir::AliasAnalysis::MayAlias ||
        AS->resultKind == fcmlir::AliasAnalysis::NoAlias) {
      LLVM_DEBUG(llvm::errs().indent(2) << "confused aa\n");
      return false;
    }

    // memref is defined inside the loop nest.
    // Not sure how to handle this!
    auto memref = storeOp.getPointer();
    if (!op.isDefinedOutsideOfLoop(memref)) {
      LLVM_DEBUG(llvm::errs().indent(2) << "pointer defined in loop\n");
      return false;
    }

    // if memref is not alloca, then return.
    if (!llvm::dyn_cast_or_null<FC::AllocaOp>(memref.getDefiningOp())) {
      LLVM_DEBUG(llvm::errs().indent(2) << "Non aloca op\n");
      return false;
    }

    assert(AS->resultKind == fcmlir::AliasAnalysis::MustAlias);
    assert(AS->memorySet.size() == 1);
    for (auto user : memref.getUsers()) {
      LLVM_DEBUG(llvm::errs().indent(2)
                 << "Checking the user " << *user << "\n");
      if (user == storeOp) {
        continue;
      }

      // If the use is not a Load then we
      // do not know the data flow.
      if (!llvm::isa<FC::FCLoadOp>(user)) {
        LLVM_DEBUG(llvm::errs().indent(2) << "user is not load\n");
        return false;
      }

      // All user should be inside loop nest.
      if (!op.getOperation()->isAncestor(user)) {
        LLVM_DEBUG(llvm::errs().indent(2) << "User is not in nest\n");
        return false;
      }

      // Store should dominate all other loads.
      if (!DT.dominates(storeOp, user)) {
        LLVM_DEBUG(llvm::errs().indent(2) << "User doesn't dominate\n");
        return false;
      }
    }

    LLVM_DEBUG(llvm::errs().indent(2) << "Safe to promote\n");
    return true;
  }

  void simplifyMemRefsInLoopNest(FC::DoOp op) {
    fcmlir::AliasSetTracker AST;
    AST.add(op.region());

    mlir::DominanceInfo DT(op);

    llvm::SmallVector<FC::FCStoreOp, 2> storesToPromote;

    // TODO: optimize algorithm.
    op.walk([&](FC::FCStoreOp storeOp) {
      if (canPromoteMemToReg(storeOp, op, AST, DT))
        storesToPromote.push_back(storeOp);
    });

    // Now replace all the stores to Reg and delete the
    // alloc operation.
    for (auto storeOp : storesToPromote) {
      auto value = storeOp.getValueToStore();
      auto memref = storeOp.getPointer();
      auto allocOp = llvm::cast<FC::AllocaOp>(memref.getDefiningOp());
      for (auto user : storeOp.getPointer().getUsers()) {
        if (user == storeOp)
          continue;
        auto loadOp = llvm::cast<FC::FCLoadOp>(user);
        loadOp.replaceAllUsesWith(value);
        loadOp.erase();
      }
      storeOp.erase();
      allocOp.erase();
    }
  }

  virtual void runOnOperation() {
    auto F = getOperation();

    // TODO : Enable when required
    bool simplifyMemRefs = false;

    LLVM_DEBUG(llvm::errs()
               << "Simplify memops pass on function " << F.getName() << "\n");
    llvm::SmallVector<FC::DoOp, 2> topLevelLoops;
    F.walk([&](FC::DoOp op) {
      // Work on top level loop nests only.
      if (op.getParentOfType<FC::DoOp>()) {
        return;
      }
      topLevelLoops.push_back(op);
    });

    for (auto loop : topLevelLoops) {
      // Hoist loop invariant loads.
      hoistInvariantLoads(loop);

      // Try to promote some memory patterns to register.
      if (simplifyMemRefs)
        simplifyMemRefsInLoopNest(loop);
    }
  }
}; // namespace mlir
} // namespace mlir

/// Create a LoopTransform pass.
std::unique_ptr<mlir::Pass> createSimplifyLoopMemOperations() {
  return std::make_unique<mlir::SimplifyLoopMemOps>();
}

static mlir::PassRegistration<mlir::SimplifyLoopMemOps>
    pass("simplify-loop-mem", "Pass to hoist Loop invariant loads");
