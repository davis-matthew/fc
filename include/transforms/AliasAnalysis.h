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
//===--- AliasAnalysis.h - AliasAnalysis based utilities ------------------===//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_ALIAS_ANALYSIS_H
#define MLIR_TRANSFORMS_ALIAS_ANALYSIS_H

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Function.h"

#include "dialect/FC/FCOps.h"

#include <map>
#include <memory>
#include <vector>

namespace fcmlir {

struct AliasAnalysis {
  enum AliasResult { NoAlias, MayAlias, MustAlias };
  static AliasResult alias(mlir::Operation *Src, mlir::Operation *Dst);
  static AliasResult alias(mlir::Value srcPtr, mlir::Value dstPtr);
};

struct AliasSet {
  enum AccessKind { NoAccess = 0, RefAccess = 1, ModAccess = 2, ModRefAccess };
  void addAccess(mlir::Value ptr, AccessKind kind);
  AliasAnalysis::AliasResult aliases(mlir::Value ptr, AccessKind kind);
  void setAccessKind(AccessKind kind) { access |= kind; }

  bool isRef() const { return access & RefAccess; }
  bool isMod() const { return access & ModAccess; }

  llvm::SmallPtrSet<mlir::Value, 2> memorySet;
  unsigned access : 2;
  AliasAnalysis::AliasResult resultKind;
};

struct AliasSetTracker {
private:
  // Holds the actual alias set.
  std::map<mlir::Value, AliasSet *> pointerMap;
  llvm::SmallVector<AliasSet *, 2> aliasSets;

public:
  ~AliasSetTracker() {
    for (auto AS : aliasSets) {
      delete AS;
    }
  }
  void add(mlir::Region &region) {
    for (auto &BB : region) {
      add(&BB);
    }
  }
  void add(mlir::Block *block) {
    for (auto &op : *block) {
      add(&op);
    }
  }

  void add(mlir::Operation *op);

  void add(mlir::Value ptr, AliasSet::AccessKind kind);

  AliasSet *getAliasSetFor(mlir::Operation *op);
};

} // namespace fcmlir

#endif
