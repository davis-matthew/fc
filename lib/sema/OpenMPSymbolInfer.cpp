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
#include "AST/ASTContext.h"
#include "AST/ASTPass.h"

#include "AST/ParseTreeBuilder.h"
#include "AST/StmtVisitor.h"
#include "AST/SymbolTable.h"
#include "AST/Type.h"
#include "common/Diagnostics.h"
#include "sema/ExpansionUtils.h"
#include "sema/Intrinsics.h"

using namespace fc;
using namespace ast;
namespace fc {

// TODO : Redesign this implementation
class OpenMPSymbolInfer : public StmtVisitor<OpenMPSymbolInfer, bool> {
  SymbolList symbolList;

public:
  OpenMPSymbolInfer() {}

  SymbolList getSymbolList() { return symbolList; }

  bool visitObjectName(ObjectName *objectName) override {
    symbolList.push_back((objectName->getSymbol()));
    return true;
  }

  bool visitOpenMPParallelDoStmt(OpenMPParallelDoStmt *stmt) override {
    visit(stmt->getDoStmt()->getQuadExpr());
    for (auto statemtent : stmt->getDoStmt()->getBlock()->getStmtList())
      visit(statemtent);
    return true;
  }

  bool visitDoStmt(DoStmt *stmt) override {
    visit(stmt->getQuadExpr());
    for (auto statemtent : stmt->getBlock()->getStmtList())
      visit(statemtent);
    return true;
  }

  bool visitOpenMPParallelStmt(OpenMPParallelStmt *stmt) override {
    for (auto statemtent : stmt->getBlock()->getStmtList())
      visit(statemtent);
    return true;
  }

  bool visitOpenMPSingleStmt(OpenMPSingleStmt *stmt) override {
    for (auto statemtent : stmt->getBlock()->getStmtList())
      visit(statemtent);
    return true;
  }

  bool visitOpenMPMasterStmt(OpenMPMasterStmt *stmt) override {
    for (auto statemtent : stmt->getBlock()->getStmtList())
      visit(statemtent);
    return true;
  }

  bool visitOpenMPDoStmt(OpenMPDoStmt *stmt) override {
    for (auto statemtent : stmt->getDoStmt()->getBlock()->getStmtList())
      visit(statemtent);
    return true;
  }
};

template <class T> static void updateMappedSymbol(T *stmt, SymbolList symList) {
  for (auto sym : symList) {
    stmt->insertMapedSymbol(sym);
  }
}

class OpenMPSymbolInferPass : public ASTBlockPass {
  OpenMPSymbolInfer infer;

public:
  OpenMPSymbolInferPass(ASTContext &C)
      : ASTBlockPass(C, "Symbol infer Pass"), infer() {}

  bool runOnBlock(Block *block) override {
    auto stmtList = block->getStmtList();

    for (auto stmt : stmtList) {
      Block *block = nullptr;
      if (auto ompStmt = llvm::dyn_cast<OpenMPParallelStmt>(stmt)) {
        block = ompStmt->getBlock();
      } else if (auto doStmt = llvm::dyn_cast<OpenMPParallelDoStmt>(stmt)) {
        block = doStmt->getDoStmt()->getBlock();
      }
      if (!block)
        continue;
      for (auto statemtent : block->getStmtList()) {
        if (!infer.visit(statemtent)) {
          return false;
        }
      }
      if (auto ompStmt = llvm::dyn_cast<OpenMPParallelStmt>(stmt)) {
        updateMappedSymbol(ompStmt, infer.getSymbolList());
      } else if (auto doStmt = llvm::dyn_cast<OpenMPParallelDoStmt>(stmt)) {
        updateMappedSymbol(doStmt, infer.getSymbolList());
      }
    }
    return true;
  }
};

ASTPass *createOpenMPSymbolInferPass(ASTContext &C) {
  return new OpenMPSymbolInferPass(C);
}
} // namespace fc
