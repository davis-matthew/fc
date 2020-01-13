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
//===- IteratedDominanceFrontier.cpp - Compute IDF ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compute iterated dominance frontiers using a linear time algorithm.
//
//===----------------------------------------------------------------------===//

// copied from
// https://raw.githubusercontent.com/schweitzpgi/f18/f18/lib/fir/IteratedDominanceFrontier.cpp

#include "transforms/IteratedDominanceFrontier.h"

#include "mlir/Analysis/Dominance.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include <queue>
#include <utility>

namespace fcmlir {

template <class NodeTy, bool IsPostDom>
void IDFCalculator<NodeTy, IsPostDom>::calculate(
    llvm::SmallVectorImpl<NodeTy *> &PHIBlocks) {
  // Use a priority queue keyed on dominator tree level so that inserted nodes
  // are handled from the bottom of the dominator tree upwards. We also augment
  // the level with a DFS number to ensure that the blocks are ordered in a
  // deterministic way.
  using UnsignedPair = std::pair<unsigned, unsigned>;
  using DomTreeNode = llvm::DomTreeNodeBase<NodeTy>;
  using DomTreeNodePair = std::pair<DomTreeNode *, UnsignedPair>;
  using IDFPriorityQueue =
      std::priority_queue<DomTreeNodePair,
                          llvm::SmallVector<DomTreeNodePair, 32>,
                          llvm::less_second>;
  IDFPriorityQueue PQ;

  if (DefBlocks->empty())
    return;

  DT.updateDFSNumbers();

  for (NodeTy *BB : *DefBlocks) {
    if (DomTreeNode *Node = DT.getNode(BB))
      PQ.push({Node, std::make_pair(Node->getLevel(), Node->getDFSNumIn())});
  }

  llvm::SmallVector<DomTreeNode *, 32> Worklist;
  llvm::SmallPtrSet<DomTreeNode *, 32> VisitedPQ;
  llvm::SmallPtrSet<DomTreeNode *, 32> VisitedWorklist;

  while (!PQ.empty()) {
    DomTreeNodePair RootPair = PQ.top();
    PQ.pop();
    DomTreeNode *Root = RootPair.first;
    unsigned RootLevel = RootPair.second.first;

    // Walk all dominator tree children of Root, inspecting their CFG edges with
    // targets elsewhere on the dominator tree. Only targets whose level is at
    // most Root's level are added to the iterated dominance frontier of the
    // definition set.

    Worklist.clear();
    Worklist.push_back(Root);
    VisitedWorklist.insert(Root);

    while (!Worklist.empty()) {
      DomTreeNode *Node = Worklist.pop_back_val();
      NodeTy *BB = Node->getBlock();
      // Succ is the successor in the direction we are calculating IDF, so it is
      // successor for IDF, and predecessor for Reverse IDF.
      auto DoWork = [&](NodeTy *Succ) {
        DomTreeNode *SuccNode = DT.getNode(Succ);

        const unsigned SuccLevel = SuccNode->getLevel();
        if (SuccLevel > RootLevel)
          return;

        if (!VisitedPQ.insert(SuccNode).second)
          return;

        NodeTy *SuccBB = SuccNode->getBlock();
        if (useLiveIn && !LiveInBlocks->count(SuccBB))
          return;

        PHIBlocks.emplace_back(SuccBB);
        if (!DefBlocks->count(SuccBB))
          PQ.push(std::make_pair(
              SuccNode, std::make_pair(SuccLevel, SuccNode->getDFSNumIn())));
      };

      llvm::SmallVector<mlir::Block *, 2> succs;
      for (auto &op : *BB) {
        for (auto &region : op.getRegions()) {
          if (region.empty())
            continue;
          succs.push_back(&region.front());
        }
      }
      for (auto Succ : BB->getSuccessors()) {
        succs.push_back(Succ);
      }

      for (auto *Succ : succs)
        DoWork(Succ);

      for (auto DomChild : *Node) {
        if (VisitedWorklist.insert(DomChild).second)
          Worklist.push_back(DomChild);
      }
    }
  }
}

template class IDFCalculator<mlir::Block, false>;

} // namespace fcmlir