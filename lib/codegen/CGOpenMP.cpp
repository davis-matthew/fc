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
#include "AST/Statements.h"
#include "AST/StmtOpenMP.h"
#include "AST/SymbolTable.h"
#include "AST/Type.h"
#include "codegen/CGASTHelper.h"
#include "codegen/CodeGen.h"
#include "common/Debug.h"
#include "dialect/OpenMPOps/OpenMPOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "llvm/Support/CommandLine.h"

using namespace fc;
using namespace ast;

bool CodeGen::emitOpenMPParallelStmt(OpenMPParallelStmt *stmt) {
  auto oldRegion = context.currRegion;
  auto mlirloc = getLoc(stmt->getSourceLoc());

  SymbolList mapedSymbol;
  stmt->getMapedSymbols(mapedSymbol);

  llvm::SmallVector<mlir::Value, 2> mappedValues;
  for (auto sym : mapedSymbol) {
    mappedValues.push_back(context.getMLIRValueFor(sym->getName()));
  }

  auto ompOp = builder.create<OMP::ParallelOp>(mlirloc, mappedValues);

  context.currRegion = &ompOp.region();

  if (context.currRegion->empty()) {
    builder.createBlock(context.currRegion, context.currRegion->end());
  }

  auto block = &context.currRegion->front();

  // Start emitting the loop body.
  builder.setInsertionPointToStart(block);
  if (!emitExectubaleConstructList(stmt->getBlock()->getStmtList())) {
    return false;
  }

  builder.create<OMP::OpenMPTerminatorOp>(mlirloc);
  builder.setInsertionPointAfter(ompOp);
  context.currRegion = oldRegion;
  return true;
}
