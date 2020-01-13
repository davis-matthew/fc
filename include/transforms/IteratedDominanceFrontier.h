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
//===- IteratedDominanceFrontier.h - Calculate IDF --------------*- C++ -*-===//

// copied from
// https://raw.githubusercontent.com/schweitzpgi/f18/f18/include/fir/Analysis/IteratedDominanceFrontier.h

#ifndef MLIR_ANALYSIS_IDF_H
#define MLIR_ANALYSIS_IDF_H

#include "mlir/Analysis/Dominance.h"
#include "mlir/IR/Block.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Block;
class DominanceInfo;
} // namespace mlir

namespace fcmlir {

/// Determine the iterated dominance frontier, given a set of defining
/// blocks, and optionally, a set of live-in blocks.
///
/// In turn, the results can be used to place phi nodes.
///
/// This algorithm is a linear time computation of Iterated Dominance Frontiers,
/// pruned using the live-in set.
/// By default, liveness is not used to prune the IDF computation.
/// The template parameters should be either BasicBlock* or Inverse<BasicBlock
/// *>, depending on if you want the forward or reverse IDF.
template <class NodeTy, bool IsPostDom> class IDFCalculator {
public:
  IDFCalculator(mlir::DominanceInfo &DT) : DT(DT), useLiveIn(false) {}

  /// Give the IDF calculator the set of blocks in which the value is
  /// defined.  This is equivalent to the set of starting blocks it should be
  /// calculating the IDF for (though later gets pruned based on liveness).
  ///
  /// Note: This set *must* live for the entire lifetime of the IDF calculator.
  void setDefiningBlocks(const llvm::SmallPtrSetImpl<NodeTy *> &Blocks) {
    DefBlocks = &Blocks;
  }

  /// Give the IDF calculator the set of blocks in which the value is
  /// live on entry to the block.   This is used to prune the IDF calculation to
  /// not include blocks where any phi insertion would be dead.
  ///
  /// Note: This set *must* live for the entire lifetime of the IDF calculator.

  void setLiveInBlocks(const llvm::SmallPtrSetImpl<NodeTy *> &Blocks) {
    LiveInBlocks = &Blocks;
    useLiveIn = true;
  }

  /// Reset the live-in block set to be empty, and tell the IDF
  /// calculator to not use liveness anymore.
  void resetLiveInBlocks() {
    LiveInBlocks = nullptr;
    useLiveIn = false;
  }

  /// Calculate iterated dominance frontiers
  ///
  /// This uses the linear-time phi algorithm based on DJ-graphs mentioned in
  /// the file-level comment.  It performs DF->IDF pruning using the live-in
  /// set, to avoid computing the IDF for blocks where an inserted PHI node
  /// would be dead.
  void calculate(llvm::SmallVectorImpl<NodeTy *> &IDFBlocks);

private:
  mlir::DominanceInfo &DT;
  bool useLiveIn;
  const llvm::SmallPtrSetImpl<NodeTy *> *LiveInBlocks;
  const llvm::SmallPtrSetImpl<NodeTy *> *DefBlocks;
};

typedef IDFCalculator<mlir::Block, false> ForwardIDFCalculator;

} // namespace fcmlir
#endif