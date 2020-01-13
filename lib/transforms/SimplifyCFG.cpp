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
//===- SimplifyCFG.cpp - Remove dead Blocks/ simplify Blocks---------------===//
//
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

using namespace std;

namespace mlir {

struct SimplifyCFG : public FunctionPass<SimplifyCFG> {
  virtual void runOnFunction() {
    auto F = getFunction();

    llvm::SmallVector<Region *, 32> regions;
    F.walk([&](Operation *op) {
      if (op->getNumRegions() > 0) {
        mlir::simplifyRegions(op->getRegions());
      }
    });
    return;
  }
};
} // namespace mlir

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createSimplifyCFGPass() {
  return std::make_unique<mlir::SimplifyCFG>();
}

static mlir::PassRegistration<mlir::SimplifyCFG>
    pass("simplify-cfg", "Pass to convert loop.for to affine.for");
