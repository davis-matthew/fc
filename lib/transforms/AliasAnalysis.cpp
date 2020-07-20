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

#include "transforms/AliasAnalysis.h"

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

#include "dialect/FC/FCOps.h"

#include "common/Debug.h"

using namespace fcmlir;
using namespace mlir;

#define DEBUG_TYPE "aa"

using AliasResult = AliasAnalysis::AliasResult;
using AliasKind = AliasSet::AccessKind;

static void
getPointerOperands(mlir::Operation *op,
                   llvm::SmallVectorImpl<mlir::Value> &pointerOperands,
                   AliasSet::AccessKind &kind) {

  // TODO : move to switch based impl
  if (auto load = dyn_cast<FC::FCLoadOp>(op)) {
    pointerOperands.push_back(load.getPointer());
    kind = AliasSet::RefAccess;
    return;
  }
  if (auto store = dyn_cast<FC::FCStoreOp>(op)) {
    pointerOperands.push_back(store.getPointer());
    kind = AliasSet::ModAccess;
    return;
  }
  if (auto read = dyn_cast<FC::ReadOp>(op)) {
    kind = AliasSet::ModAccess;
    for (auto op : read.getOperands()) {
      if (op.getType().isa<FC::RefType>()) {
        pointerOperands.push_back(op);
      }
    }
    return;
  }
  if (auto write = dyn_cast<FC::WriteOp>(op)) {
    kind = AliasSet::RefAccess;
    for (auto op : write.getOperands()) {
      if (op.getType().isa<FC::RefType>()) {
        pointerOperands.push_back(op);
      }
    }
    return;
  }
  if (auto print = dyn_cast<FC::PrintOp>(op)) {
    kind = AliasSet::RefAccess;
    for (auto op : print.getOperands()) {
      if (op.getType().isa<FC::RefType>()) {
        pointerOperands.push_back(op);
      }
    }
    return;
  }
  if (auto call = dyn_cast<mlir::CallOp>(op)) {
    kind = AliasSet::ModAccess;
    for (auto op : call.getOperands()) {
      if (op.getType().isa<FC::RefType>()) {
        pointerOperands.push_back(op);
      }
    }
    return;
  }
  kind = AliasSet::NoAccess;
  return;
}

AliasResult AliasAnalysis::alias(mlir::Operation *Src, mlir::Operation *Dst) {
  llvm::SmallVector<mlir::Value, 1> operands;
  AliasSet::AccessKind kind;
  getPointerOperands(Src, operands, kind);
  // Non memory operation.
  if (kind == AliasSet::NoAccess) {
    return AliasResult::NoAlias;
  }
  assert(operands.size() == 1);
  auto srcPtr = operands.front();
  operands.clear();
  getPointerOperands(Dst, operands, kind);
  // Non memory operation.
  if (kind == AliasSet::NoAccess) {
    return AliasResult::NoAlias;
  }
  assert(operands.size() == 1);
  auto dstPtr = operands.front();
  return alias(srcPtr, dstPtr);
}

AliasResult AliasAnalysis::alias(mlir::Value srcPtr, mlir::Value dstPtr) {
  auto srcDefOp = srcPtr.getDefiningOp();
  auto dstDefOp = dstPtr.getDefiningOp();

  // Check if both are memory allocations.
  auto srcAllocOp = dyn_cast_or_null<FC::AllocaOp>(srcDefOp);
  auto dstAllocOp = dyn_cast_or_null<FC::AllocaOp>(dstDefOp);
  if (srcAllocOp && dstAllocOp) {
    return (srcDefOp == dstDefOp) ? MustAlias : NoAlias;
  }

  // Check if both are block arguments.
  auto srcBlockArg = srcPtr.dyn_cast<mlir::BlockArgument>();
  auto dstBlockArg = dstPtr.dyn_cast<mlir::BlockArgument>();

  if ((srcBlockArg && !srcBlockArg.getOwner()->isEntryBlock()) ||
      (dstBlockArg && !dstBlockArg.getOwner()->isEntryBlock())) {
    // This needs further analysis.
    // TODO: This cannot be generated in the current IR.
    return MayAlias;
  }

  // Fortran arguments doesn't alias with each other.
  if (srcBlockArg && dstBlockArg) {
    return (srcBlockArg == dstBlockArg) ? MustAlias : NoAlias;
  }

  // TODO : check for recursive function calls?
  // Arguments and alloca operations cannot alias with each other.
  if ((srcAllocOp && dstBlockArg) || (srcBlockArg && dstAllocOp)) {
    return NoAlias;
  }

  // Check if both are global allocations.
  auto srcAddrOfOp = dyn_cast_or_null<FC::AddressOfOp>(srcDefOp);
  auto dstAddrOfOp = dyn_cast_or_null<FC::AddressOfOp>(dstDefOp);

  // Globals do not alias with others.
  if (srcAddrOfOp && dstAddrOfOp) {
    return (srcAddrOfOp == dstAddrOfOp) ? MustAlias : NoAlias;

    auto srcGlobal = srcAddrOfOp.getGlobal();
    auto dstGlobal = dstAddrOfOp.getGlobal();
    return (srcGlobal == dstGlobal) ? MustAlias : NoAlias;
  }

  // Globals do not alias with local stack allocations.
  if ((srcAddrOfOp && dstAllocOp) || (dstAddrOfOp && srcAllocOp)) {
    return NoAlias;
  }

  auto srcGetRef = dyn_cast_or_null<FC::GetElementRefOp>(srcDefOp);
  auto dstGetRef = dyn_cast_or_null<FC::GetElementRefOp>(dstDefOp);

  // Variables from other scopes, parent function / module.
  if (srcGetRef && dstGetRef) {
    return (srcGetRef.getSymRef() == dstGetRef.getSymRef()) ? MustAlias
                                                            : NoAlias;
  }

  // Other scope variables doesn't alias with locals. But it may alias
  // with the function arguments!
  if ((srcGetRef && dstAllocOp) || (dstGetRef && srcAllocOp)) {
    return NoAlias;
  }

  // TODO: Handle FC.arrayele
  return MayAlias;
}

void AliasSet::addAccess(mlir::Value ptr, AccessKind kind) {
  memorySet.insert(ptr);
  setAccessKind(kind);
}

// TODO: optimize the algorithm.
void AliasSetTracker::add(mlir::Value ptr, AliasSet::AccessKind accessKind) {
  assert(ptr);
  auto ASI = pointerMap.find(ptr);
  if (ASI != pointerMap.end()) {
    (*ASI).second->setAccessKind(accessKind);
    return;
  }

  // If it is not found, iterate over existing sets to see if it aliases.
  for (auto AS : aliasSets) {
    auto currASptr = *(AS->memorySet.begin());
    auto aliasResult = AliasAnalysis::alias(currASptr, ptr);
    if (aliasResult == AliasAnalysis::NoAlias) {
      continue;
    }
    AS->addAccess(ptr, accessKind);
    AS->resultKind = aliasResult;
    pointerMap[ptr] = AS;
    return;
  }

  // Not found the alias set yet. Create one and return.
  AliasSet *set = new AliasSet();
  set->addAccess(ptr, accessKind);
  set->resultKind = AliasAnalysis::MustAlias;
  pointerMap[ptr] = set;
  aliasSets.push_back(set);
}

// TODO: optimize the algorithm.
void AliasSetTracker::add(Operation *op) {

  // First traverse all the regions in the Op.
  for (auto &nestedRegion : op->getRegions()) {
    add(nestedRegion);
  }

  // TODO: Find other instructions which affects alias set.
  llvm::SmallVector<mlir::Value, 2> pointerOperands;
  AliasSet::AccessKind accessKind;
  getPointerOperands(op, pointerOperands, accessKind);
  for (auto ptr : pointerOperands)
    add(ptr, accessKind);
}

AliasSet *AliasSetTracker::getAliasSetFor(mlir::Operation *op) {
  llvm::SmallVector<mlir::Value, 2> operands;
  AliasSet::AccessKind kind;
  getPointerOperands(op, operands, kind);
  if (operands.empty()) {
    return nullptr;
  }
  assert(operands.size() == 1);
  auto PMI = pointerMap.find(operands.front());
  assert(PMI != pointerMap.end());
  return PMI->second;
}
