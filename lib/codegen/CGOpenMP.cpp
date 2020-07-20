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
#include "AST/Statements.h"
#include "AST/StmtOpenMP.h"
#include "AST/SymbolTable.h"
#include "AST/Type.h"
#include "codegen/CGASTHelper.h"
#include "codegen/CodeGen.h"
#include "common/Debug.h"
#include "dialect/OpenMP/OpenMPOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "llvm/Support/CommandLine.h"

using namespace fc;
using namespace ast;

bool CodeGen::emitOpenMPParallelStmt(OpenMPParallelStmt *stmt) {
  auto oldRegion = context.currRegion;
  auto mlirloc = getLoc(stmt->getSourceLoc());

  SymbolSet mapedSymbol;
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

bool CodeGen::emitOpenMPSingleStmt(OpenMPSingleStmt *stmt) {
  auto oldRegion = context.currRegion;
  auto mlirloc = getLoc(stmt->getSourceLoc());

  auto ompOp = builder.create<OMP::SingleOp>(mlirloc);

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

// TODO : Most code in this function is repeated code from emitFCDoLoop.
//        Generalize these two functions
bool CodeGen::emitOpenMPDoStmt(OpenMPDoStmt *stmt) {
  auto oldRegion = context.currRegion;

  auto doStmt = stmt->getDoStmt();
  auto expr = doStmt->getQuadExpr();
  auto mlirloc = getLoc(doStmt->getSourceLoc());

  // Create a simple do loop without any expression.
  if (!expr) {
    llvm_unreachable("Can not emit omp loop without lb , ub, step");
    return true;
  }

  // set dovar value to init.
  auto indVarsymbol =
      dyn_cast<fc::ObjectName>(expr->getOperand(0))->getSymbol();

  auto indvar = context.symbolMap[indVarsymbol->getName()];
  auto ivTy = cgHelper->getMLIRTypeFor(indVarsymbol->getType());
  auto initVal = emitExpression(expr->getOperand(1));
  auto endVal = emitExpression(expr->getOperand(2));
  auto step = emitExpression(expr->getOperand(3));

  llvm::SmallVector<mlir::Value, 2> args{indvar, initVal, endVal, step};
  auto forop = builder.create<OMP::OmpDoOp>(mlirloc, args);
  auto block = forop.getBody();

  // Replace the uses of indvar variable with SSA indvar value.
  context.symbolMap[indVarsymbol->getName()] = forop.getIndVar();
  context.currRegion = &forop.region();

  // Start emitting the loop body.
  builder.setInsertionPointToStart(block);
  if (!emitExectubaleConstructList(doStmt->getBlock()->getStmtList())) {
    return false;
  }

  // Emit at the end of loop body.
  builder.create<OMP::OpenMPTerminatorOp>(mlirloc);
  builder.setInsertionPointAfter(forop);
  auto finalVal = builder.create<mlir::AddIOp>(
      mlirloc, emitCastExpr(endVal, ivTy), emitCastExpr(step, ivTy));
  emitStoreInstruction(finalVal, indvar);

  // Restore the use of indvars back to the indvar variable instead of SSA
  // value.
  context.symbolMap[indVarsymbol->getName()] = indvar;
  context.currRegion = oldRegion;
  return true;
}

bool CodeGen::emitOpenMPMasterStmt(OpenMPMasterStmt *stmt) {
  auto oldRegion = context.currRegion;
  auto mlirloc = getLoc(stmt->getSourceLoc());

  auto ompOp = builder.create<OMP::MasterOp>(mlirloc);

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

bool CodeGen::emitOpenMPParallelDoStmt(OpenMPParallelDoStmt *stmt) {
  auto oldRegion = context.currRegion;

  auto doStmt = stmt->getDoStmt();
  auto expr = doStmt->getQuadExpr();
  auto mlirloc = getLoc(doStmt->getSourceLoc());

  // Create a simple do loop without any expression.
  if (!expr) {
    llvm_unreachable("Can not emit omp loop without lb , ub, step");
    return true;
  }

  // set dovar value to init.
  auto indVarsymbol =
      dyn_cast<fc::ObjectName>(expr->getOperand(0))->getSymbol();

  auto indvar = context.symbolMap[indVarsymbol->getName()];
  auto initVal = emitExpression(expr->getOperand(1));
  auto endVal = emitExpression(expr->getOperand(2));
  auto step = emitExpression(expr->getOperand(3));

  auto I32 = builder.getIntegerType(32);
  auto I32Ref = FC::RefType::get(I32);
  auto initAlloca = createAlloca(I32Ref, cgHelper->getTempUniqueName());
  auto endAlloca = createAlloca(I32Ref, cgHelper->getTempUniqueName());
  auto stepAlloca = createAlloca(I32Ref, cgHelper->getTempUniqueName());
  emitStoreInstruction(emitCastExpr(initVal, I32), initAlloca);
  emitStoreInstruction(emitCastExpr(endVal, I32), endAlloca);
  emitStoreInstruction(emitCastExpr(step, I32), stepAlloca);

  SymbolSet mapedSymbols;
  stmt->getMapedSymbols(mapedSymbols);

  llvm::SmallVector<mlir::Value, 2> args{indvar, initAlloca, endAlloca,
                                         stepAlloca};
  for (auto sym : mapedSymbols) {
    if (sym->getName() == indVarsymbol->getName())
      continue;
    args.push_back(context.getMLIRValueFor(sym->getName()));
  }

  auto forop = builder.create<OMP::ParallelDoOp>(mlirloc, args);
  auto block = forop.getBody();

  // Replace the uses of indvar variable with SSA indvar value.
  context.symbolMap[indVarsymbol->getName()] = forop.getIndVar();
  context.currRegion = &forop.region();

  // Start emitting the loop body.
  builder.setInsertionPointToStart(block);
  if (!emitExectubaleConstructList(doStmt->getBlock()->getStmtList())) {
    return false;
  }

  builder.create<OMP::OpenMPTerminatorOp>(mlirloc);
  builder.setInsertionPointAfter(forop);

  // Restore the use of indvars back to the indvar variable instead of SSA
  // value.
  context.symbolMap[indVarsymbol->getName()] = indvar;
  context.currRegion = oldRegion;
  return true;
}
