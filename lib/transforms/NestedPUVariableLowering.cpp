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
//===- NestedPUVariableLowering.cpp ---------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// All implicitly captured variables are explictly passed as arguments in this
// pass. This is part of high level to low level MLIR lowering.
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

#include <map>
#include <set>

using namespace std;

namespace mlir {

using FuncList = llvm::SmallVector<FC::FCFuncOp, 2>;
using UseList = llvm::SmallVector<FC::AddressOfOp, 4>;
using ArgList = llvm::SmallPtrSet<mlir::Value, 8>;

// TODO: move to util function.
static mlir::Operation *getOpForSymRef(ModuleOp module,
                                       mlir::SymbolRefAttr symRef) {
  mlir::SymbolTable table(module);
  mlir::Operation *op;
  for (auto ref : llvm::reverse(symRef.getNestedReferences())) {
    op = table.lookup(ref.getValue());
    table = mlir::SymbolTable(op);
    assert(op);
  }
  op = table.lookup(symRef.getRootReference());
  assert(op);
  return op;
}

struct NestedPUVariableLowering : public ModulePass<NestedPUVariableLowering> {
private:
  ModuleOp module;

  void collectParentVariablesUsed(FC::FCFuncOp child, FC::FCFuncOp parent,
                                  ArgList &list) {
    child.walk([&](FC::GetElementRefOp op) {
      auto symref = op.getSymRef();
      auto parentAlloca = getOpForSymRef(module, symref);
      assert(parentAlloca);
      if (parentAlloca->getParentOfType<FC::FCFuncOp>() != parent)
        return;
      assert(llvm::isa<FC::AllocaOp>(parentAlloca) ||
             llvm::isa<FC::CaptureArgOp>(parentAlloca));
      assert(parentAlloca->getNumResults() == 1);
      list.insert(parentAlloca->getResult(0));
    });
  }

  void replaceAccessAndCalls(FC::FCFuncOp childFunc, FC::FCFuncOp parent,
                             llvm::DenseMap<mlir::Value, mlir::Value> &argMap,
                             llvm::SmallVectorImpl<mlir::Value> &argList) {

    // Replace child function uses.
    childFunc.walk([&](Operation *childOp) {
      // Do not walk nested functions.
      if (childOp->getParentOfType<FC::FCFuncOp>() != childFunc) {
        return;
      }
      if (auto op = llvm::dyn_cast<FC::GetElementRefOp>(childOp)) {
        auto symref = op.getSymRef();
        auto parentAlloca = getOpForSymRef(module, symref);
        assert(parentAlloca);
        if (parentAlloca->getParentOfType<FC::FCFuncOp>() != parent)
          return;
        op.replaceAllUsesWith(argMap[parentAlloca->getResult(0)]);
        op.erase();
        return;
      }
      if (auto callOp = llvm::dyn_cast<FC::FCCallOp>(childOp)) {
        auto symref = callOp.getCallee();
        auto calledFunc = getOpForSymRef(module, symref);
        if (parent != calledFunc->getParentOfType<FC::FCFuncOp>())
          return;
        auto callee = callOp.getCallee();
        llvm::SmallVector<mlir::Value, 2> currOperands(callOp.getArgOperands());
        currOperands.append(argList.begin(), argList.end());
        OpBuilder builder(callOp.getContext());
        builder.setInsertionPointAfter(callOp);
        llvm::SmallVector<Type, 2> resultTypes(callOp.getResultTypes());
        auto newCallOp = builder.create<FC::FCCallOp>(
            callOp.getLoc(), callee, resultTypes, currOperands);
        callOp.replaceAllUsesWith(newCallOp.getResults());
        callOp.erase();
        return;
      }
    });
  }
  void lowerNestedFunctionsIn(FC::FCFuncOp parent, FuncList &children) {
    ArgList list;
    llvm::SmallVector<mlir::Value, 4> parentArgValList;
    // Collect all the parent scope variables used.
    FuncList newChildren;
    for (auto child : children) {
      collectParentVariablesUsed(child, parent, list);
    }
    if (list.empty()) {
      return;
    }
    llvm::SmallVector<Type, 2> newArgTypes;
    for (auto arg : list) {
      newArgTypes.push_back(arg.getType());
      parentArgValList.push_back(arg);
    }

    for (auto child : children) {
      auto funcType = child.getType();
      llvm::SmallVector<Type, 2> inputs{funcType.getInputs().begin(),
                                        funcType.getInputs().end()};
      inputs.append(newArgTypes.begin(), newArgTypes.end());
      OpBuilder builder(module.getContext());
      auto newFuncType = builder.getFunctionType(inputs, funcType.getResults());
      auto oldName = child.getName();
      child.setName("old");
      builder.setInsertionPointAfter(child);
      auto newFunc =
          builder.create<FC::FCFuncOp>(child.getLoc(), oldName, newFuncType);
      auto &region = child.body();
      newFunc.getBody().takeBody(region);

      llvm::SmallVector<mlir::Value, 4> newArgs;
      auto args = newFunc.front().addArguments(newArgTypes);
      newArgs.append(args.begin(), args.end());

      llvm::DenseMap<mlir::Value, mlir::Value> allocaArgMap;
      for (auto argIndex : llvm::enumerate(list)) {
        allocaArgMap[argIndex.value()] = newArgs[argIndex.index()];
      }

      // Replace all the acesses and also fix all the nested function calls.
      replaceAccessAndCalls(newFunc, parent, allocaArgMap, newArgs);
      child.erase();
    }

    // Now replace all the calls in parent function to use alloca.
    llvm::DenseMap<mlir::Value, mlir::Value> dummyMap;
    replaceAccessAndCalls(parent, parent, dummyMap, parentArgValList);
  }

public:
  virtual void runOnModule() {
    auto M = getModule();
    module = M;
    // Collect all higher parent-child functions.
    std::map<FC::FCFuncOp, FuncList> funcMap;
    // TODO: optimize.
    M.walk([&](FC::FCFuncOp op) {
      auto parent = op.getParentOfType<FC::FCFuncOp>();
      if (parent) {
        if (funcMap.find(parent) != funcMap.end()) {
          funcMap[parent].push_back(op);
          return;
        }
        funcMap[parent] = {op};
        return;
      }
      funcMap[parent] = {};
    });
    for (auto &mapEle : funcMap) {
      if (mapEle.second.empty())
        continue;
      lowerNestedFunctionsIn(mapEle.first, mapEle.second);

      // Remove all the "implitcly captured" tags.
      auto parentOp = mapEle.first;
      parentOp.walk([&](FC::AllocaOp op) {
        op.setCaptured(mlir::BoolAttr::get(false, module.getContext()));
      });
    }
  }
};

} // namespace mlir

std::unique_ptr<mlir::Pass> createNestedPUVariableLoweringPass() {
  return std::make_unique<mlir::NestedPUVariableLowering>();
}

static mlir::PassRegistration<mlir::NestedPUVariableLowering>
    pass("lower-nested-func-mem", "Pass to lower nested function variables");
