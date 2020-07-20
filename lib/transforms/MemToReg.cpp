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
//===- MemToReg.cpp - Generalized mem to reg pass for MLIR dialects     ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// copied from
// https://github.com/schweitzpgi/f18/blob/f18/lib/fir/Transforms/MemToReg.cpp
// and made some modifications.

#include "transforms/IteratedDominanceFrontier.h"

#include "mlir/Analysis/Dominance.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>
#include <vector>

#include "dialect/FC/FCOps.h"

using DominatorTree = mlir::DominanceInfo;

/// A generalized version of a mem-to-reg pass suitable for use with an MLIR
/// dialect. This code was ported from the LLVM project. MLIR differs with its
/// use of block arguments rather than PHI nodes, etc.

namespace fcmlir {

bool isAllocaPromotable(FC::AllocaOp &ae) {
  if (!ae.getType().cast<FC::RefType>().getEleTy().isIntOrIndexOrFloat()) {
    return false;
  }

  if (auto attr = ae.getAttr("alloc_kind")) {
    if (attr.cast<StringAttr>().getValue() == "static")
      return false;
  }

  if (auto attr = ae.getAttrOfType<BoolAttr>("captured"))
    if (attr.getValue())
      return false;

  for (auto &use : ae.getResult().getUses()) {
    auto *op = use.getOwner();
    if (op->getParentOfType<FC::DoOp>()) {
      return false;
    }
    if (auto load = mlir::dyn_cast<FC::FCLoadOp>(op)) {
    } else if (auto stor = mlir::dyn_cast<FC::FCStoreOp>(op)) {
      if (stor.getOperand(0).getDefiningOp() == op) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
} // namespace fcmlir

struct AllocaInfo {
  llvm::SmallVector<mlir::Block *, 32> definingBlocks;
  llvm::SmallVector<mlir::Block *, 32> usingBlocks;

  mlir::Operation *onlyStore;
  mlir::Block *onlyBlock;
  bool onlyUsedInOneBlock;

  void clear() {
    definingBlocks.clear();
    usingBlocks.clear();
    onlyStore = nullptr;
    onlyBlock = nullptr;
    onlyUsedInOneBlock = true;
  }

  /// Scan the uses of the specified alloca, filling in the AllocaInfo used
  /// by the rest of the pass to reason about the uses of this alloca.
  void analyzeAlloca(FC::AllocaOp &AI) {
    clear();

    // As we scan the uses of the alloca instruction, keep track of stores,
    // and decide whether all of the loads and stores to the alloca are within
    // the same basic block.
    for (auto UI = AI.getResult().use_begin(), E = AI.getResult().use_end();
         UI != E;) {
      auto *User = UI->getOwner();
      UI++;

      if (auto SI = mlir::dyn_cast<FC::FCStoreOp>(User)) {
        // Remember the basic blocks which define new values for the alloca
        definingBlocks.push_back(SI.getOperation()->getBlock());
        onlyStore = SI.getOperation();
      } else {
        auto LI = mlir::cast<FC::FCLoadOp>(User);
        // Otherwise it must be a load instruction, keep track of variable
        // reads.
        usingBlocks.push_back(LI.getOperation()->getBlock());
      }

      if (onlyUsedInOneBlock) {
        if (!onlyBlock)
          onlyBlock = User->getBlock();
        else if (onlyBlock != User->getBlock())
          onlyUsedInOneBlock = false;
      }
    }
  }
};

struct RenamePassData {
  using ValVector = std::vector<mlir::Value>;

  RenamePassData(mlir::Block *b, mlir::Block *p, const ValVector &v)
      : BB(b), Pred(p), Values(v) {}
  RenamePassData(const RenamePassData &) = delete;
  RenamePassData &operator=(const RenamePassData &) = delete;
  RenamePassData(RenamePassData &&) = default;
  ~RenamePassData() = default;

  mlir::Block *BB;
  mlir::Block *Pred;
  ValVector Values;
};

struct LargeBlockInfo {
  using INMap = llvm::DenseMap<mlir::Operation *, unsigned>;
  INMap instNumbers;

  static bool isInterestingInstruction(mlir::Operation &I) {
    if (mlir::isa<FC::FCLoadOp>(I)) {
      if (auto *op = I.getOperand(0).getDefiningOp())
        return mlir::isa<FC::AllocaOp>(op);
    } else if (mlir::isa<FC::FCStoreOp>(I)) {
      if (auto *op = I.getOperand(1).getDefiningOp())
        return mlir::isa<FC::AllocaOp>(op);
    }
    return false;
  }

  unsigned getInstructionIndex(Operation *oper) {
    // has it already been numbered?
    INMap::iterator it = instNumbers.find(oper);
    if (it != instNumbers.end())
      return it->second;

    // No. search for the oper
    auto *block = oper->getBlock();
    unsigned num = 0u;
    for (auto &o : block->getOperations())
      if (isInterestingInstruction(o))
        instNumbers[&o] = num++;

    it = instNumbers.find(oper);
    assert(it != instNumbers.end() && "operation not in block?");
    return it->second;
  }

  void deleteValue(Operation *oper) { instNumbers.erase(oper); }
  void clear() { instNumbers.clear(); }
};

struct MemToReg : public mlir::OperationPass<MemToReg, FC::FCFuncOp> {
  explicit MemToReg() {}

  std::vector<FC::AllocaOp> allocas;
  DominatorTree *domTree = nullptr;
  mlir::OpBuilder *builder = nullptr;

  /// Contains a stable numbering of basic blocks to avoid non-deterministic
  /// behavior.
  llvm::DenseMap<mlir::Block *, unsigned> BBNumbers;

  /// Reverse mapping of Allocas.
  llvm::DenseMap<mlir::Operation *, unsigned> allocaLookup;

  /// The set of basic blocks the renamer has already visited.
  llvm::SmallPtrSet<mlir::Block *, 16> Visited;

  llvm::DenseMap<std::pair<mlir::Block *, mlir::Operation *>, unsigned>
      BlockArgs;
  llvm::DenseMap<std::pair<mlir::Block *, unsigned>, unsigned> argToAllocaMap;

  llvm::SmallVector<mlir::Value, 2> undefs;

  bool rewriteSingleStoreAlloca(FC::AllocaOp &AI, AllocaInfo &Info,
                                LargeBlockInfo &LBI) {
    FC::FCStoreOp onlyStore(mlir::cast<FC::FCStoreOp>(Info.onlyStore));
    mlir::Block *StoreBB = Info.onlyStore->getBlock();
    int StoreIndex = -1;

    // Clear out usingBlocks.  We will reconstruct it here if needed.
    Info.usingBlocks.clear();

    for (auto UI = AI.getResult().use_begin(), E = AI.getResult().use_end();
         UI != E;) {
      auto *UserInst = UI->getOwner();
      UI++;

      if (mlir::dyn_cast<FC::FCStoreOp>(UserInst))
        continue;

      auto LI = mlir::cast<FC::FCLoadOp>(UserInst);

      // Okay, if we have a load from the alloca, we want to replace it with the
      // only value stored to the alloca.  We can do this if the value is
      // dominated by the store.  If not, we use the rest of the MemToReg
      // machinery to insert the phi nodes as needed.
      if (LI.getOperation()->getBlock() == StoreBB) {
        // If we have a use that is in the same block as the store, compare
        // the indices of the two instructions to see which one came first. If
        // the load came before the store, we can't handle it.
        if (StoreIndex == -1)
          StoreIndex = LBI.getInstructionIndex(onlyStore);

        if (unsigned(StoreIndex) > LBI.getInstructionIndex(LI)) {
          // Can't handle this load, bail out.
          Info.usingBlocks.push_back(StoreBB);
          continue;
        }
      } else if (!domTree->dominates(StoreBB, LI.getOperation()->getBlock())) {
        // If the load and store are in different blocks, use BB dominance to
        // check their relationships.  If the store doesn't dom the use, bail
        // out.
        Info.usingBlocks.push_back(LI.getOperation()->getBlock());
        continue;
      }

      // Otherwise, we *can* safely rewrite this load.
      mlir::Value ReplVal = onlyStore.getOperand(0);
      // If the replacement value is the load, this must occur in unreachable
      // code.
      if (ReplVal == LI.getResult())
        ReplVal = createUndef(LI.getLoc(), LI.getType());

      LI.replaceAllUsesWith(ReplVal);
      LI.erase();
      LBI.deleteValue(LI);
    }

    // Finally, after the scan, check to see if the store is all that is left.
    if (!Info.usingBlocks.empty())
      return false; // If not, we'll have to fall back for the remainder.

    // Remove the (now dead) store and alloca.
    Info.onlyStore->erase();

    AI.erase();
    return true;
  }

  bool promoteSingleBlockAlloca(FC::AllocaOp AI, AllocaInfo &Info,
                                LargeBlockInfo &LBI) {
    // Walk the use-def list of the alloca, getting the locations of all stores.
    using StoresByIndexTy =
        llvm::SmallVector<std::pair<unsigned, FC::FCStoreOp>, 64>;
    StoresByIndexTy StoresByIndex;

    for (auto U = AI.getResult().use_begin(), E = AI.getResult().use_end();
         U != E; U++)
      if (auto SI = llvm::dyn_cast<FC::FCStoreOp>(U->getOwner()))
        StoresByIndex.emplace_back(LBI.getInstructionIndex(SI), SI);

    // Sort the stores by their index, making it efficient to do a lookup with a
    // binary search.
    llvm::sort(StoresByIndex, llvm::less_first());

    // Walk all of the loads from this alloca, replacing them with the nearest
    // store above them, if any.
    for (auto UI = AI.getResult().use_begin(), E = AI.getResult().use_end();
         UI != E;) {
      auto LI = mlir::dyn_cast<FC::FCLoadOp>(UI->getOwner());
      UI++;
      if (!LI)
        continue;

      unsigned LoadIdx = LBI.getInstructionIndex(LI);

      // Find the nearest store that has a lower index than this load.
      auto I = llvm::lower_bound(
          StoresByIndex,
          std::make_pair(LoadIdx, static_cast<FC::FCStoreOp>(nullptr)),
          llvm::less_first());

      if (I == StoresByIndex.begin()) {
        if (StoresByIndex.empty()) {
          // If there are no stores, the load takes the undef value.
          auto undef = createUndef(LI.getLoc(), LI.getType());
          LI.replaceAllUsesWith(undef);
        } else {
          // There is no store before this load, bail out (load may be affected
          // by the following stores - see main comment).
          return false;
        }
      } else {
        // Otherwise, there was a store before this load, the load takes its
        // value. Note, if the load was marked as nonnull we don't want to lose
        // that information when we erase it. So we preserve it with an assume.
        mlir::Value ReplVal = std::prev(I)->second.getOperand(0);

        // If the replacement value is the load, this must occur in unreachable
        // code.
        if (ReplVal == LI)
          ReplVal = createUndef(LI.getLoc(), LI.getType());

        LI.replaceAllUsesWith(ReplVal);
      }

      LI.erase();
      LBI.deleteValue(LI);
    }

    // Remove the (now dead) stores and alloca.
    while (!AI.use_empty()) {
      auto ae = AI.getResult();
      for (auto ai = ae.user_begin(), E = ae.user_end(); ai != E; ai++)
        if (FC::FCStoreOp SI = mlir::dyn_cast<FC::FCStoreOp>(*ai)) {
          SI.erase();
          LBI.deleteValue(SI);
        }
    }

    AI.erase();
    return true;
  }

  void
  computeLiveInBlocks(FC::AllocaOp ae, AllocaInfo &Info,
                      const llvm::SmallPtrSetImpl<mlir::Block *> &DefBlocks,
                      llvm::SmallPtrSetImpl<mlir::Block *> &liveInBlks) {
    auto *AI = ae.getOperation();
    // To determine liveness, we must iterate through the predecessors of blocks
    // where the def is live.  Blocks are added to the worklist if we need to
    // check their predecessors.  Start with all the using blocks.
    llvm::SmallVector<mlir::Block *, 64> LiveInBlockWorklist(
        Info.usingBlocks.begin(), Info.usingBlocks.end());

    // If any of the using blocks is also a definition block, check to see if
    // the definition occurs before or after the use.  If it happens before the
    // use, the value isn't really live-in.
    for (unsigned i = 0, e = LiveInBlockWorklist.size(); i != e; ++i) {
      mlir::Block *BB = LiveInBlockWorklist[i];
      if (!DefBlocks.count(BB))
        continue;

      // Okay, this is a block that both uses and defines the value.  If the
      // first reference to the alloca is a def (store), then we know it isn't
      // live-in.
      for (mlir::Block::iterator I = BB->begin();; ++I) {
        if (FC::FCStoreOp SI = mlir::dyn_cast<FC::FCStoreOp>(I)) {
          if (SI.getOperand(1).getDefiningOp() != AI)
            continue;

          // We found a store to the alloca before a load.  The alloca is not
          // actually live-in here.
          LiveInBlockWorklist[i] = LiveInBlockWorklist.back();
          LiveInBlockWorklist.pop_back();
          --i;
          --e;
          break;
        }

        if (auto LI = mlir::dyn_cast<FC::FCLoadOp>(I))
          // Okay, we found a load before a store to the alloca.  It is actually
          // live into this block.
          if (LI.getPointer().getDefiningOp() == AI)
            break;
      }
    }

    // Now that we have a set of blocks where the phi is live-in, recursively
    // add their predecessors until we find the full region the value is live.
    while (!LiveInBlockWorklist.empty()) {
      mlir::Block *BB = LiveInBlockWorklist.pop_back_val();

      // The block really is live in here, insert it into the set.  If already
      // in the set, then it has already been processed.
      if (!liveInBlks.insert(BB).second)
        continue;

      // Since the value is live into BB, it is either defined in a predecessor
      // or live into it to.  Add the preds to the worklist unless they are a
      // defining block.
      for (mlir::Block *P : BB->getPredecessors()) {
        // The value is not live into a predecessor if it defines the value.
        if (DefBlocks.count(P))
          continue;

        // Otherwise it is, add to the worklist.
        LiveInBlockWorklist.push_back(P);
      }
    }
  }

  bool addBlockArgument(mlir::Block *BB, FC::AllocaOp &Alloca,
                        unsigned allocaNum) {
    auto *ae = Alloca.getOperation();
    auto key = std::make_pair(BB, ae);
    auto argNoIter = BlockArgs.find(key);
    if (argNoIter != BlockArgs.end())
      return false;
    auto argNo = BB->getNumArguments();
    BB->addArgument(Alloca.getType().getEleTy());
    BlockArgs[key] = argNo;
    argToAllocaMap[std::make_pair(BB, argNo)] = allocaNum;
    return true;
  }

  void initOperands(std::vector<mlir::Value> &opers, mlir::Location &&loc,
                    mlir::Block *dest, unsigned size, unsigned ai,
                    mlir::Value val, Operation::operand_range &&oldOpers) {
    unsigned i = 0;
    for (auto v : oldOpers)
      opers[i++] = v;

    // we must fill additional args with temporary undef values
    for (; i < size; ++i) {
      if (i == ai)
        continue;
      auto opTy = dest->getArgument(i).getType();
      auto typedUndef = createUndef(loc, opTy);
      opers[i] = typedUndef;
    }
    opers[ai] = val;
  }

  static void eraseIfNoUse(mlir::Value val) {
    if (val.use_begin() == val.use_end()) {
      val.getDefiningOp()->erase();
    }
  }

  /// Set the incoming value on the branch side for the `ai`th block argument
  void setParam(mlir::Block *blk, unsigned ai, mlir::Value val,
                mlir::Block *target, unsigned size) {
    auto *term = blk->getTerminator();
    if (auto br = mlir::dyn_cast<mlir::BranchOp>(term)) {
      if (br.getNumOperands() <= ai) {
        // construct a new BranchOp to replace term
        std::vector<mlir::Value> opers(size);
        auto *dest = br.getDest();
        builder->setInsertionPoint(term);
        initOperands(opers, br.getLoc(), dest, size, ai, val, br.getOperands());
        builder->create<mlir::BranchOp>(br.getLoc(), dest, opers);
        br.erase();
      } else {
        auto oldParam = br.getOperand(ai);
        br.setOperand(ai, val);
        eraseIfNoUse(oldParam);
      }
    } else if (auto cond = mlir::dyn_cast<mlir::CondBranchOp>(term)) {
      if (target == cond.getTrueDest()) {
        if (cond.getNumTrueOperands() <= ai) {
          // construct a new CondBranchOp to replace term
          std::vector<mlir::Value> opers(size);
          auto *dest = cond.getTrueDest();
          builder->setInsertionPoint(term);
          initOperands(opers, cond.getLoc(), dest, size, ai, val,
                       cond.getTrueOperands());
          auto c = cond.getCondition();
          auto *othDest = cond.getFalseDest();
          std::vector<mlir::Value> othOpers(cond.false_operand_begin(),
                                            cond.false_operand_end());
          builder->create<mlir::CondBranchOp>(cond.getLoc(), c, dest, opers,
                                              othDest, othOpers);
          cond.erase();
        } else {
          auto oldParam = cond.getTrueOperand(ai);
          cond.setTrueOperand(ai, val);
          eraseIfNoUse(oldParam);
        }
      } else {
        if (cond.getNumFalseOperands() <= ai) {
          // construct a new CondBranchOp to replace term
          std::vector<mlir::Value> opers(size);
          auto *dest = cond.getFalseDest();
          builder->setInsertionPoint(term);
          initOperands(opers, cond.getLoc(), dest, size, ai, val,
                       cond.getFalseOperands());
          auto c = cond.getCondition();
          auto *othDest = cond.getTrueDest();
          std::vector<mlir::Value> othOpers(cond.true_operand_begin(),
                                            cond.true_operand_end());
          builder->create<mlir::CondBranchOp>(cond.getLoc(), c, othDest,
                                              othOpers, dest, opers);
          cond.erase();
        } else {
          auto oldParam = cond.getFalseOperand(ai);
          cond.setFalseOperand(ai, val);
          eraseIfNoUse(oldParam);
        }
      }
    } else {
      assert(false && "unhandled terminator");
    }
  }

  inline static void addValue(RenamePassData::ValVector &vector,
                              RenamePassData::ValVector::size_type size,
                              mlir::Value value) {
    if (vector.size() < size + 1)
      vector.resize(size + 1);
    vector[size] = value;
  }

  mlir::Value createUndef(mlir::Location loc, mlir::Type type) {
    auto undef = builder->create<FC::UndefOp>(loc, type).getResult();
    return undef;
  }
  /// Recursively traverse the CFG of the function, renaming loads and
  /// stores to the allocas which we are promoting.
  ///
  /// IncomingVals indicates what value each Alloca contains on exit from the
  /// predecessor block Pred.
  void renamePass(mlir::Block *BB, mlir::Block *Pred,
                  RenamePassData::ValVector &IncomingVals,
                  std::vector<RenamePassData> &Worklist) {
  NextIteration:
    // Does this block take arguments?

    if ((!BB->hasNoPredecessors()) && (BB->getNumArguments() > 0)) {
      // add the values from block `Pred` to the argument list in the proper
      // positions
      for (unsigned ai = 0, AI = BB->getNumArguments(); ai != AI; ++ai) {
        auto allocaNo = argToAllocaMap[std::make_pair(BB, ai)];
        setParam(Pred, ai, IncomingVals[allocaNo], BB, AI);
        // use the block argument, not the live def in the pred block
        addValue(IncomingVals, allocaNo, BB->getArgument(ai));
      }
    }

    // Don't revisit blocks.
    if (!Visited.insert(BB).second)
      return;

    mlir::Block::iterator II = BB->begin();
    while (true) {
      if (II == BB->end())
        break;
      mlir::Operation &opn = *II;
      II++;

      if (auto LI = mlir::dyn_cast<FC::FCLoadOp>(opn)) {
        auto *srcOpn = LI.getPointer().getDefiningOp();
        if (!srcOpn)
          continue;

        auto Src = mlir::dyn_cast<FC::AllocaOp>(srcOpn);
        if (!Src)
          continue;

        llvm::DenseMap<mlir::Operation *, unsigned>::iterator AI =
            allocaLookup.find(srcOpn);
        if (AI == allocaLookup.end())
          continue;

        mlir::Value V = IncomingVals[AI->second];

        // Anything using the load now uses the current value.
        LI.replaceAllUsesWith(V);
        LI.erase();
      } else if (auto SI = mlir::dyn_cast<FC::FCStoreOp>(opn)) {
        auto *dstOpn = SI.getOperand(1).getDefiningOp();
        if (!dstOpn)
          continue;

        // Delete this instruction and mark the name as the current holder of
        // the value
        auto Dest = mlir::dyn_cast<FC::AllocaOp>(dstOpn);
        if (!Dest)
          continue;

        llvm::DenseMap<mlir::Operation *, unsigned>::iterator ai =
            allocaLookup.find(dstOpn);
        if (ai == allocaLookup.end())
          continue;

        // what value were we writing?
        unsigned AllocaNo = ai->second;
        addValue(IncomingVals, AllocaNo, SI.getOperand(0));
        SI.erase();
      }
    }

    // 'Recurse' to our successors.
    auto I = BB->succ_begin();
    auto E = BB->succ_end();
    if (I == E)
      return;

    // Keep track of the successors so we don't visit the same successor twice
    llvm::SmallPtrSet<mlir::Block *, 8> VisitedSuccs;

    // Handle the first successor without using the worklist.
    VisitedSuccs.insert(*I);
    Pred = BB;
    BB = *I;
    ++I;

    for (; I != E; ++I)
      if (VisitedSuccs.insert(*I).second)
        Worklist.emplace_back(*I, Pred, IncomingVals);
    goto NextIteration;
  }

  void doPromotion() {
    auto F = this->getOperation();
    std::vector<FC::AllocaOp> aes;
    AllocaInfo info;
    LargeBlockInfo lbi;
    ForwardIDFCalculator IDF(*domTree);

    assert(!allocas.empty());

    for (unsigned allocaNum = 0, End = allocas.size(); allocaNum != End;
         ++allocaNum) {
      auto ae = allocas[allocaNum];
      assert(ae.getParentOfType<FC::FCFuncOp>() == F);
      if (ae.use_empty()) {
        ae.erase();
        continue;
      }
      info.analyzeAlloca(ae);
      builder->setInsertionPointToStart(&F.front());
      if (info.definingBlocks.size() == 1)
        if (rewriteSingleStoreAlloca(ae, info, lbi))
          continue;
      if (info.onlyUsedInOneBlock)
        if (promoteSingleBlockAlloca(ae, info, lbi))
          continue;

      // If we haven't computed a numbering for the BB's in the function, do
      // so now.
      if (BBNumbers.empty()) {
        unsigned ID = 0;

        F.walk([&](mlir::Operation *op) {
          if (BBNumbers.find(op->getBlock()) != BBNumbers.end())
            return;
          BBNumbers[op->getBlock()] = ID++;
        });
      }

      // Keep the reverse mapping of the 'Allocas' array for the rename pass.
      allocaLookup[allocas[allocaNum].getOperation()] = allocaNum;

      // At this point, we're committed to promoting the alloca using IDF's,
      // and the standard SSA construction algorithm.  Determine which blocks
      // need PHI nodes and see if we can optimize out some work by avoiding
      // insertion of dead phi nodes.

      // Unique the set of defining blocks for efficient lookup.
      llvm::SmallPtrSet<mlir::Block *, 32> DefBlocks(
          info.definingBlocks.begin(), info.definingBlocks.end());

      // Determine which blocks the value is live in.  These are blocks which
      // lead to uses.
      llvm::SmallPtrSet<mlir::Block *, 32> liveInBlks;
      computeLiveInBlocks(ae, info, DefBlocks, liveInBlks);

      // At this point, we're committed to promoting the alloca using IDF's,
      // and the standard SSA construction algorithm.  Determine which blocks
      // need phi nodes and see if we can optimize out some work by avoiding
      // insertion of dead phi nodes.
      IDF.setLiveInBlocks(liveInBlks);
      IDF.setDefiningBlocks(DefBlocks);
      llvm::SmallVector<mlir::Block *, 32> PHIBlocks;
      IDF.calculate(PHIBlocks);
      llvm::sort(PHIBlocks, [this](mlir::Block *A, mlir::Block *B) {
        return BBNumbers.find(A)->second < BBNumbers.find(B)->second;
      });

      for (mlir::Block *BB : PHIBlocks)
        addBlockArgument(BB, ae, allocaNum);

      aes.push_back(ae);
    }

    std::swap(allocas, aes);
    if (allocas.empty())
      return;

    lbi.clear();

    // Set the incoming values for the basic block to be null values for all
    // of the alloca's.  We do this in case there is a load of a value that
    // has not been stored yet.  In this case, it will get this null value.
    RenamePassData::ValVector Values(allocas.size());
    for (unsigned i = 0, e = allocas.size(); i != e; ++i)
      Values[i] =
          createUndef(allocas[i].getLoc(), allocas[i].getType().getEleTy());

    // Walks all basic blocks in the function performing the SSA rename
    // algorithm and inserting the phi nodes we marked as necessary
    std::vector<RenamePassData> renameWorklist;
    renameWorklist.emplace_back(&F.front(), nullptr, Values);
    do {
      RenamePassData RPD(std::move(renameWorklist.back()));
      renameWorklist.pop_back();
      // renamePass may add new worklist entries.
      renamePass(RPD.BB, RPD.Pred, RPD.Values, renameWorklist);
    } while (!renameWorklist.empty());

    // The renamer uses the Visited set to avoid infinite loops.  Clear it
    // now.
    Visited.clear();

    // Remove the allocas themselves from the function.
    for (auto aa : allocas) {
      mlir::Operation *A = aa.getOperation();
      // If there are any uses of the alloca instructions left, they must be
      // in unreachable basic blocks that were not processed by walking the
      // dominator tree. Just delete the users now.
      if (!A->use_empty()) {
        auto undef = createUndef(aa.getLoc(), aa.getType());
        aa.replaceAllUsesWith(undef);
      }
      aa.erase();
    }
  }

  /// run the MemToReg pass on the FIR dialect
  virtual void runOnOperation() override {
    auto f = this->getOperation();
    if (f.isExternal())
      return;

    auto &entry = f.front();
    auto bldr = mlir::OpBuilder(f.getBody());

    domTree = &this->getAnalysis<DominatorTree>();
    builder = &bldr;

    while (true) {
      allocas.clear();

      for (auto &op : entry)
        if (FC::AllocaOp ae = mlir::dyn_cast<FC::AllocaOp>(op))
          if (isAllocaPromotable(ae))
            allocas.push_back(ae);

      if (allocas.empty())
        break;

      doPromotion();
    }

    // Remove all the unused undefs created.
    for (auto undef : undefs) {
      if (undef.use_empty()) {
        undef.getDefiningOp()->erase();
      }
    }
    undefs.clear();
    domTree = nullptr;
    builder = nullptr;
  }
}; // namespace fcmlir

} // namespace fcmlir

std::unique_ptr<Pass> createMemToRegPass() {
  return std::make_unique<fcmlir::MemToReg>();
}
