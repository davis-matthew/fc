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
#include "dialect/FCOps/FCOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "llvm/Support/CommandLine.h"

using namespace fc;
using namespace ast;
using namespace llvm;

// TODO: Works only for full range array section.
bool CodeGen::emitArraySectionStore(Expr *lhs, Expr *rhs,
                                    mlir::Location mlirloc) {

  auto lhsArrSec = llvm::dyn_cast<ArraySection>(lhs);
  // auto rhsArrSec = llvm::dyn_cast<ArraySection>(rhs);

  assert(lhsArrSec);

  // only full range is handled for now.
  // assert(rhsArrSec && lhsArrSec->isFullRange() && rhsArrSec->isFullRange());

  auto rhsVal = emitExpression(rhs);
  auto lhsVal = emitExpression(lhs, true);
  builder.create<FC::FCStoreOp>(mlirloc, rhsVal, lhsVal);
  return true;
}

bool CodeGen::emitAssignment(AssignmentStmt *stmt) {
  auto rhs = stmt->getRHS();
  auto lhs = stmt->getLHS();
  auto mlirloc = getLoc(rhs->getSourceLoc());

  if (llvm::isa<ArraySection>(lhs)) {
    return emitArraySectionStore(lhs, rhs, mlirloc);
  }

  auto rhsVal = emitExpression(rhs);

  if (rhsVal.getType().isa<mlir::IndexType>() &&
      Type::getCoreElementType(lhs->getType())->isInt32Ty()) {
    rhsVal = builder.create<mlir::IndexCastOp>(
        mlirloc, rhsVal, mlir::IntegerType::get(32, &mlirContext));
  }
  if (auto arrEle = llvm::dyn_cast<ArrayElement>(lhs)) {
    auto Alloca = context.symbolMap[arrEle->getSymbol()->getName()];
    assert(Alloca);
    FC::SubscriptRangeList subs;
    for (auto sub : arrEle->getSubscriptList()) {
      auto tempSub = emitExpression(sub);
      subs.push_back(FC::SubscriptRange(castToIndex(tempSub)));
    }
    auto storeOp = builder.create<FC::FCStoreOp>(getLoc(stmt->getSourceLoc()),
                                                 rhsVal, Alloca, subs);
    storeOp.setAttr("name",
                    builder.getStringAttr(arrEle->getSymbol()->getName()));
    return true;
  }

  auto lhsVal = emitExpression(lhs, true);
  assert(lhsVal);

  auto storeOp = builder.create<FC::FCStoreOp>(getLoc(stmt->getSourceLoc()),
                                               rhsVal, lhsVal);
  if (auto objName = dyn_cast<ObjectName>(lhs)) {
    storeOp.setAttr("name", builder.getStringAttr(objName->getName()));
  }

  return true;
}

static bool isLhs(Expr *expr) {
  if (auto arrayEle = llvm::dyn_cast<ArrayElement>(expr)) {
    auto arrayTy = llvm::dyn_cast<fc::ArrayType>(arrayEle->getType());
    assert(arrayTy);
    if (arrayTy->getNumDims() == arrayEle->getNumIndices())
      return false;
    return true;
  }

  if (auto arraySec = llvm::dyn_cast<ArraySection>(expr)) {
    if (arraySec->isFullRange())
      return true;
    llvm_unreachable("Partial array section should be replaced in sema");
  }
  return false;
}

// Emits fc.print
bool CodeGen::emitPrintStmt(PrintStmt *stmt) {
  if (stmt->getNumOperands() == 0)
    return true;
  llvm::SmallVector<mlir::Value, 2> exprValList;
  auto mlirloc = getLoc(stmt->getSourceLoc());

  llvm::BitVector isString;
  for (auto stmt : stmt->getOperands()) {
    auto expr = llvm::dyn_cast<Expr>(stmt);
    assert(expr);
    isString.push_back(expr->getType()->isStringArrTy());
    mlir::Value exprVal = nullptr;
    exprVal = emitExpression(expr, isLhs(expr));
    assert(exprVal);
    exprValList.push_back(exprVal);
  }

  auto printOp = builder.create<FC::PrintOp>(mlirloc, exprValList);
  // String needs better handling.
  printOp.setAttr("arg_info", FC::StringInfoAttr::get(&mlirContext, isString));

  return true;
}

bool CodeGen::emitInternalWriteStmt(WriteStmt *stmt) {
  auto unit = stmt->getUnit();
  auto iostat = stmt->getIostat();
  mlir::Location mlirloc = getLoc(stmt->getSourceLoc());
  mlir::Value iostatValue = nullptr;
  mlir::Value zero = nullptr;

  if (iostat) {
    iostatValue = emitExpression(iostat, true);
    zero = builder.create<mlir::ConstantIntOp>(mlirloc, 0, 32);
  }

  ArrayType *arrayTy = llvm::dyn_cast<ArrayType>(unit->getType());
  if (!arrayTy)
    return false;

  if (!arrayTy->getElementTy()->isCharacterTy() &&
      !arrayTy->getElementTy()->isStringCharTy())
    return false;

  ExprList exprList = stmt->getExprList();

  if (exprList.size() == 1) {
    auto expr = exprList[0];
    if (!expr->getType()->isIntegralTy()) {
      llvm_unreachable("Not handled!");
    }

    assert(expr->getType()->isIntegralTy());
    auto exprVal = emitExpression(expr, false);
    auto unitExpr = emitExpression(unit, true);
    builder.create<FC::ItoSOp>(mlirloc, unitExpr, exprVal);

    // Currently storing zero.
    if (iostatValue) {
      builder.create<mlir::StoreOp>(mlirloc, zero, iostatValue);
    }
    return true;
  }

  bool isLHS;
  llvm::BitVector isString;
  llvm::SmallVector<mlir::Value, 2> exprValList;
  for (auto expr : exprList) {
    assert(expr);
    isLHS = isLhs(expr);
    auto exprVal = emitExpression(expr, isLHS);
    assert(exprVal);
    exprValList.push_back(exprVal);
    isString.push_back(expr->getType()->isStringArrTy());
  }

  auto unitExpr = emitExpression(unit, true);
  auto sprintfOp =
      builder.create<FC::SprintfOp>(mlirloc, unitExpr, exprValList);
  sprintfOp.setAttr("arg_info",
                    FC::StringInfoAttr::get(&mlirContext, isString));

  // Currently storing zero.
  if (iostatValue) {
    builder.create<mlir::StoreOp>(mlirloc, zero, iostatValue);
  }
  return true;
}

bool CodeGen::emitWriteStmt(WriteStmt *writeStmt) {
  if (writeStmt->getNumOperands() == 0)
    return true;

  llvm::SmallVector<mlir::Value, 2> exprValList;
  mlir::Location mlirloc = getLoc(writeStmt->getSourceLoc());

  mlir::Value unitVal = nullptr;

  // TODO: what should we do about iostat?
  // how to handle spaceList ?

  if (writeStmt->getUnit())
    if (emitInternalWriteStmt(writeStmt))
      return true;

  if (Expr *unit = writeStmt->getUnit()) {
    unitVal = emitExpression(unit);
  } else {
    unitVal = builder.create<mlir::ConstantIntOp>(mlirloc, 6, 32);
  }

  llvm::SmallVector<int, 2> spaceList;
  writeStmt->getSpaceList(spaceList);
  assert(spaceList.size() == writeStmt->getExprList().size());

  llvm::BitVector isString;
  for (fc::Expr *expr : writeStmt->getExprList()) {
    mlir::Value exprVal = nullptr;
    bool stringType = false;
    auto ArrTy = dyn_cast<ArrayType>(expr->getType());
    if (ArrTy && ArrTy->getElementTy()->isStringCharTy()) {
      stringType = true;
    }
    if (isa<fc::ArraySection>(expr))
      exprVal = emitExpression(expr, true);
    else
      exprVal = emitExpression(expr);
    assert(exprVal);
    if (exprVal.getType().isInteger(8))
      stringType = false;
    isString.push_back(stringType);
    exprValList.push_back(exprVal);
  }

  auto writeOp = builder.create<FC::WriteOp>(mlirloc, unitVal, exprValList);
  writeOp.setAttr("arg_info", FC::StringInfoAttr::get(&mlirContext, isString));
  writeOp.setAttr("space_list", builder.getI32ArrayAttr(spaceList));

  return true;
}

bool CodeGen::emitInternalReadStmt(ReadStmt *stmt) {
  auto unit = stmt->getUnit();
  if (!unit)
    return false;
  ArrayType *arrayTy = llvm::dyn_cast<ArrayType>(unit->getType());
  if (!arrayTy)
    return false;
  if (!arrayTy->getElementTy()->isCharacterTy() &&
      !arrayTy->getElementTy()->isStringCharTy())
    return false;

  ExprList exprList = stmt->getExprList();
  assert(exprList.size() == 1);
  auto mlirLoc = getLoc(stmt->getSourceLoc());

  auto expr = exprList[0];
  auto exprVal = emitExpression(expr, true);
  assert(exprVal);
  auto unitExpr = emitExpression(unit, true);
  assert(unitExpr);

  // TODO Handle more generically.
  //      Remove duplicate code
  if (auto arraySec = llvm::dyn_cast<ArraySection>(expr)) {
    assert(arraySec->isFullRange());
    builder.create<FC::StoIAOp>(mlirLoc, unitExpr, exprVal);
    return true;
  }

  if (!expr->getType()->isIntegralTy()) {
    auto arrayEle = llvm::dyn_cast<ArrayElement>(expr);
    assert(arrayEle && "Not handled");
    if (!arrayEle->getElementType()->isIntegralTy())
      llvm_unreachable("Not handled");
  }

  auto I32 = mlir::IntegerType::get(32, &mlirContext);
  auto stoi = builder.create<FC::StoIOp>(mlirLoc, I32, unitExpr);
  builder.create<FC::FCStoreOp>(mlirLoc, stoi.getResult(), exprVal);
  // assert(stoi);
  return true;
}

bool CodeGen::emitReadStmt(ReadStmt *readStmt) {
  if (readStmt->getNumOperands() == 0)
    return true;

  if (emitInternalReadStmt(readStmt))
    return true;

  llvm::SmallVector<mlir::Value, 2> exprValList;
  mlir::Location mlirloc = getLoc(readStmt->getSourceLoc());

  mlir::Value unitVal = nullptr;

  // TODO: what should we do about iostat?
  if (Expr *unit = readStmt->getUnit()) {
    unitVal = emitExpression(unit);
  } else {
    unitVal = builder.create<mlir::ConstantIntOp>(mlirloc, 5, 32);
  }

  llvm::BitVector isString;
  for (fc::Expr *expr : readStmt->getExprList()) {
    isString.push_back(expr->getType()->isStringArrTy());
    mlir::Value exprVal = emitExpression(expr, /* isLHS */ true);
    assert(exprVal);
    exprValList.push_back(exprVal);
  }

  auto I32 = mlir::IntegerType::get(32, theModule->getContext());
  auto readOp = builder.create<FC::ReadOp>(mlirloc, I32, unitVal, exprValList);
  readOp.setAttr("arg_info", FC::StringInfoAttr::get(&mlirContext, isString));

  if (auto iostat = readStmt->getIostat()) {
    auto statValue = emitExpression(iostat, true);
    assert(statValue);
    builder.create<FC::FCStoreOp>(mlirloc, readOp.getResult(), statValue);
  }

  return true;
}

bool CodeGen::emitCloseStmt(CloseStmt *closeStmt) {
  auto unitExpr = closeStmt->getUnit();
  auto unit = emitExpression(unitExpr);
  assert(unit);
  auto I32 = mlir::IntegerType::get(32, theModule->getContext());
  mlir::Location mlirloc = getLoc(closeStmt->getSourceLoc());
  auto closeOp = builder.create<FC::CloseOp>(mlirloc, I32, unit);
  assert(closeOp);

  if (auto iostat = closeStmt->getIostat()) {
    auto statValue = emitExpression(iostat, true);
    assert(statValue);
    builder.create<FC::FCStoreOp>(mlirloc, closeOp.getResult(), statValue);
  }
  return true;
}

bool CodeGen::emitOpenStmt(OpenStmt *openStmt) {
  if (openStmt->getNumOperands() == 0)
    return true;

  llvm::SmallVector<mlir::Value, 2> exprValList;
  mlir::Location mlirloc = getLoc(openStmt->getSourceLoc());

  Expr *file = openStmt->getFile();
  Expr *unit = openStmt->getUnit();
  assert(file && unit);

  mlir::Value unitVal = emitExpression(unit);
  mlir::Value fileVal = emitExpression(file, /* isLHS */ true);

  auto I32 = mlir::IntegerType::get(32, theModule->getContext());

  auto openOp = builder.create<FC::OpenOp>(mlirloc, I32, unitVal, fileVal);
  assert(openOp);

  if (auto iostat = openStmt->getIostat()) {
    auto statValue = emitExpression(iostat, true);
    assert(statValue);
    builder.create<FC::FCStoreOp>(mlirloc, openOp.getResult(), statValue);
  }
  return true;
}

// loop.if representation of if-else statements.
bool CodeGen::emitLoopIfOperation(IfElseStmt *stmt) {

  auto &kindList = stmt->getKindList();
  auto insertPt = builder.saveInsertionPoint();
  auto numClauses = stmt->getNumOperands();
  // There is no else-if in loop.if.
  // which is handled using nested loop.if inside else region.
  for (unsigned i = 0; i < numClauses; i++) {
    auto ifPair = stmt->getIfStmt(i);
    assert(ifPair);
    auto expr = ifPair->getCondition();
    auto block = ifPair->getBlock();
    assert(block);
    mlir::Value exprVal = nullptr;

    // Emit the final else.
    if (kindList[i] == IfConstructKind::ElseKind) {
      assert(i == numClauses - 1);
      assert(!expr);
      emitExectubaleConstructList(block->getStmtList());
      break;
    }

    exprVal = emitExpression(expr);
    bool hasElse = i != numClauses - 1;
    auto ifOp = builder.create<mlir::loop::IfOp>(getLoc(stmt->getSourceLoc()),
                                                 exprVal, hasElse);
    // Emit code for if block.
    auto thenBlock = ifOp.getThenBodyBuilder().getBlock();
    builder.setInsertionPointToStart(thenBlock);
    emitExectubaleConstructList(block->getStmtList());

    // Prepare for else block.
    if (hasElse) {
      auto elseBlock = ifOp.getElseBodyBuilder().getBlock();
      builder.setInsertionPointToStart(elseBlock);
    }
  }

  builder.restoreInsertionPoint(insertPt);
  return true;
}

bool CodeGen::emitIfElseStmt(IfElseStmt *stmt) {

  // Thi is inside the structure loop like operation.
  if (context.currRegion) {
    return emitLoopIfOperation(stmt);
  }

  auto &kindList = stmt->getKindList();
  auto ExitBB = context.currFn.addBlock();

  for (unsigned i = 0; i < stmt->getNumOperands(); i++) {
    auto ifPair = stmt->getIfStmt(i);
    auto expr = ifPair->getCondition();
    auto block = ifPair->getBlock();
    mlir::Value exprVal = nullptr;
    auto ThenBB = getNewBlock(ExitBB);
    mlir::Block *ElseBlock = nullptr;
    if (kindList[i] != IfConstructKind::ElseKind) {
      exprVal = emitExpression(expr);
      ElseBlock = ExitBB;
      if (i != stmt->getNumOperands() - 1) {
        ElseBlock = getNewBlock(ExitBB);
      }
      SmallVector<mlir::Value, 2> e1, e2;
      builder.create<mlir::CondBranchOp>(getLoc(stmt->getSourceLoc()), exprVal,
                                         ThenBB, e1, ElseBlock, e2);
    } else {
      builder.create<mlir::BranchOp>(getLoc(stmt->getSourceLoc()), ThenBB);
      ElseBlock = ExitBB;
    }

    // Emit code for if block.
    context.currBB = ThenBB;
    builder.setInsertionPointToEnd(ThenBB);
    // Emit the block statements.
    emitExectubaleConstructList(block->getStmtList());
    /// Add the terminator for the then block.
    builder.create<mlir::BranchOp>(getLoc(stmt->getSourceLoc()), ExitBB);

    // Set the current BB to else block (regular block)
    context.currBB = ElseBlock;
    builder.setInsertionPointToEnd(ElseBlock);
  }
  return true;
}

mlir::Operation *CodeGen::handleCmdLineArgs(Symbol *symbol,
                                            ExprList &argsList) {
  assert(symbol->getName() == "get_command_argument");
  assert(argsList.size() == 2);
  auto pos = emitExpression(argsList[0]);
  auto mlirloc = pos.getLoc();
  auto var = emitExpression(argsList[1], true);
  return builder.create<FC::ArgvOp>(mlirloc, pos, var);
}

mlir::Value CodeGen::castToIndex(mlir::Value v) {
  if (!v || v.getType().isa<mlir::IndexType>() ||
      v.isa<mlir::BlockArgument>()) {
    return v;
  }
  auto op = v.getDefiningOp();
  if (auto constVal = dyn_cast_or_null<mlir::ConstantIntOp>(op)) {
    auto returnVal =
        builder.create<ConstantIndexOp>(v.getLoc(), constVal.getValue());
    op->replaceAllUsesWith(returnVal);
    op->erase();
    return returnVal;
  }
  return builder.create<mlir::IndexCastOp>(v.getLoc(), v,
                                           builder.getIndexType());
}

bool CodeGen::emitFCDoWhileLoop(DoWhileStmt *stmt) {
  auto expr = stmt->getLogicalExpr();
  auto mlirloc = getLoc(stmt->getSourceLoc());

  // Create a new block for logical header emission.
  auto Header = context.currFn.addBlock();

  auto Body = getNewBlock(Header);
  auto Exit = getNewBlock(Header);

  // Create direct br.
  builder.create<mlir::BranchOp>(mlirloc, Header);

  context.currBB = Header;
  builder.setInsertionPointToEnd(Header);

  // Emit loop condition expression in Header.
  auto exprVal = emitExpression(expr);
  SmallVector<mlir::Value, 2> e1, e2;
  builder.create<mlir::CondBranchOp>(mlirloc, exprVal, Body, e1, Exit, e2);

  // Emit code for loop body.
  context.currBB = Body;
  builder.setInsertionPointToEnd(Body);

  // Emit the block statements.
  emitExectubaleConstructList(stmt->getBlock()->getStmtList());

  // Create backedge to Header from body.
  builder.create<mlir::BranchOp>(mlirloc, Header);

  // Set the current BB to exit block (regular block)
  context.currBB = Exit;
  builder.setInsertionPointToEnd(Exit);
  return true;
}

bool CodeGen::emitFCDoLoop(DoStmt *stmt) {
  auto oldRegion = context.currRegion;
  auto expr = stmt->getQuadExpr();
  auto mlirloc = getLoc(stmt->getSourceLoc());
  auto attr = builder.getStringAttr(stmt->getName());

  // Create a simple do loop without any expression.
  if (!expr) {
    auto forop = builder.create<FC::DoOp>(mlirloc, attr);
    context.currRegion = &forop.region();
    auto block = forop.getBody();
    builder.setInsertionPointToStart(block);
    // context.currLoopVector.push_back(CGLoop(stmt->getName()));
    if (!emitExectubaleConstructList(stmt->getBlock()->getStmtList())) {
      return false;
    }
    builder.setInsertionPointAfter(forop);
    context.currRegion = oldRegion;
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

  // Create fc.do op.
  auto forop = builder.create<FC::DoOp>(mlirloc, attr, initVal, endVal, step);
  auto block = forop.getBody();

  // Replace the uses of indvar variable with SSA indvar value.
  context.symbolMap[indVarsymbol->getName()] = block->getArgument(0);
  context.currRegion = &forop.region();

  // Start emitting the loop body.
  builder.setInsertionPointToStart(block);
  // context.currLoopVector.push_back(CGLoop(stmt->getName()));
  if (!emitExectubaleConstructList(stmt->getBlock()->getStmtList())) {
    return false;
  }
  // context.currLoopVector.pop_back();

  // Emit at the end of loop body.
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

// TODO: For now, loops with cycle and exit are converted
// to CFG based loops.
// Loop operation with cycle/exit support is on the way.
static bool containsIllegalStmts(fc::Block *block) {
  for (auto stmt : *block) {
    switch (stmt->getStmtType()) {
    case fc::DoWhileStmtKind:
    case fc::CycleStmtKind:
    case fc::ExitStmtKind:
      return true;
    case fc::DoStmtKind: {
      auto doStmt = static_cast<DoStmt *>(stmt);
      if (!doStmt->getQuadExpr() || containsIllegalStmts(doStmt->getBlock()))
        return true;
      break;
    }
    case fc::IfElseStmtKind: {
      auto ifStmt = static_cast<IfElseStmt *>(stmt);
      auto numClauses = ifStmt->getNumOperands();
      for (unsigned i = 0; i < numClauses; i++) {
        auto ifPair = ifStmt->getIfStmt(i);
        assert(ifPair);
        if (containsIllegalStmts(ifPair->getBlock()))
          return true;
      }
    }
    default:
      break;
    }
  }
  return false;
}

bool CodeGen::emitDoStmt(DoStmt *stmt) {
  if (stmt->getQuadExpr() && !containsIllegalStmts(stmt->getBlock()))
    return emitFCDoLoop(stmt);

  auto expr = stmt->getQuadExpr();

  mlir::Value tripCount = nullptr;
  mlir::Value tripIndVar = nullptr;
  mlir::Value doVar = nullptr;
  mlir::Value stepVal = nullptr;
  bool IncrIsOne = false;
  mlir::Value initVal = nullptr, endVal = nullptr;
  mlir::Type doTy;
  if (expr) {
    auto incrExpr = expr->getOperand(3);
    if (auto constVal = llvm::dyn_cast<ConstantVal>(incrExpr)) {
      if (constVal->getInt() == 1) {
        IncrIsOne = true;
      }
    }

    // set dovar value to init.
    doVar = emitExpression(expr->getOperand(0), true);
    doTy = cgHelper->getMLIRTypeFor(expr->getOperand(0)->getType());
    initVal = emitExpression(expr->getOperand(1));
    stepVal = emitExpression(expr->getOperand(3));
    endVal = emitExpression(expr->getOperand(2));

    // set tripIndVar to zero.
    if (!IncrIsOne) {
      auto Zero = builder.create<mlir::ConstantIndexOp>(initVal.getLoc(), 0);

      tripIndVar =
          createAlloca(FC::RefType::get((initVal.getType())), "tripIndVar")
              .getResult();
      emitStoreInstruction(emitCastExpr(Zero, initVal.getType()), tripIndVar);

      auto loc = doVar.getLoc();
      auto UMinusB = builder.create<mlir::SubIOp>(loc, endVal, initVal);
      auto PlusS = builder.create<mlir::AddIOp>(loc, UMinusB, stepVal);
      tripCount = builder.create<mlir::SignedDivIOp>(loc, PlusS, stepVal);
    }

    emitStoreInstruction(emitCastExpr(initVal, doTy), doVar);
  }

  auto ExitBB = context.currFn.addBlock();

  // Create Header where loop comparison happens.
  auto Header = getNewBlock(ExitBB);

  // Body, where the loop block sits.
  auto BodyBB = getNewBlock(ExitBB);

  // Block where indvar increments happen.
  auto LatchBB = getNewBlock(ExitBB);

  fc::CGLoop *cgLoop = new CGLoop(Header, LatchBB, ExitBB, stmt->getName());
  context.currLoopVector.push_back(cgLoop);
  context.stmtLoopMap[stmt] = cgLoop;
  if (!stmt->getName().empty())
    context.nameLoopMap[stmt->getName()] = cgLoop;

  // Redirect to header.
  builder.create<mlir::BranchOp>(getLoc(stmt->getSourceLoc()), Header);

  context.currBB = Header;
  builder.setInsertionPointToEnd(Header);

  if (expr) {
    mlir::Value LoopCmp = nullptr;
    auto loc = getLoc(stmt->getSourceLoc());
    if (!IncrIsOne) {
      auto indVarVal = emitLoadInstruction(tripIndVar, "tripIndVar.load");
      // Now emit trip count condition.
      LoopCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt,
                                             indVarVal, tripCount);
    } else {
      mlir::Value doVarLoad = emitLoadInstruction(doVar, "doVar.load");
      auto One = builder.create<mlir::ConstantIndexOp>(loc, 1);
      mlir::Value finalVal = builder.create<mlir::AddIOp>(
          loc, endVal, emitCastExpr(One, endVal.getType()));
      if (doVarLoad.getType() != finalVal.getType()) {
        doVarLoad = emitCastExpr(doVarLoad, finalVal.getType());
      }
      LoopCmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt,
                                             doVarLoad, finalVal);
    }

    SmallVector<mlir::Value, 2> e1, e2;
    builder.create<mlir::CondBranchOp>(getLoc(stmt->getSourceLoc()), LoopCmp,
                                       BodyBB, e1, ExitBB, e2);

  } else {
    builder.create<mlir::BranchOp>(getLoc(stmt->getSourceLoc()), BodyBB);
  }

  // Emit Loop body now.
  context.currBB = BodyBB;
  builder.setInsertionPointToEnd(BodyBB);

  // Emit the block statements.
  emitExectubaleConstructList(stmt->getBlock()->getStmtList());

  builder.create<mlir::BranchOp>(getLoc(stmt->getSourceLoc()), LatchBB);

  // After the loop body emission, set the current loop to null.
  context.currLoopVector.pop_back();

  // Emit Loop latch now.
  context.currBB = LatchBB;
  builder.setInsertionPointToEnd(LatchBB);

  // Increment doVar by step.
  if (expr) {
    auto doVarLoad = emitLoadInstruction(doVar, "doVar.load").getResult();
    auto doVarInc =
        builder.create<mlir::AddIOp>(doVarLoad.getLoc(), doVarLoad, stepVal);
    emitStoreInstruction(emitCastExpr(doVarInc, doTy), doVar);

    if (!IncrIsOne) {
      // Increment tripIndVar by 1.
      auto One = builder.create<mlir::ConstantIntOp>(
          doVarLoad.getLoc(), 1, endVal.getType().getIntOrFloatBitWidth());
      auto indVarLoad =
          emitLoadInstruction(tripIndVar, "indvar.load").getResult();
      auto indVarInc =
          builder.create<mlir::AddIOp>(doVarLoad.getLoc(), indVarLoad, One);
      emitStoreInstruction(indVarInc, tripIndVar);
    }
  }

  // Loop back edge to Header.
  builder.create<mlir::BranchOp>(getLoc(stmt->getSourceLoc()), Header);

  // Now set continue IR dumping in Exit.
  context.currBB = ExitBB;
  builder.setInsertionPointToEnd(ExitBB);
  return true;
}

bool CodeGen::emitCallStmt(CallStmt *stmt) {
  auto symbol = stmt->getCalledFn();
  symbol = symbol->getOrigSymbol();
  auto argsList = stmt->getArgsList();
  return emitCall(symbol, argsList, true);
}

mlir::Operation *CodeGen::emitTrimCall(Symbol *symbol, ExprList &argsList) {
  assert(symbol->getName() == "trim");
  assert(argsList.size() == 1);
  auto arg = emitExpression(argsList[0], true);
  return builder.create<FC::TrimOp>(arg.getLoc(), arg);
}

// TODO: This should have been handled at AST level.
// cleanup the code.
mlir::Operation *CodeGen::handleMemCopyCall(Symbol *symbol,
                                            ExprList &argsList) {
  FC::FCCallOp Op;
  assert(symbol->getName() == "memcpy");
  assert(argsList.size() == 3);

  auto dst = emitExpression(argsList[0], true);
  auto src = emitExpression(argsList[1]);

  auto dstType = dst.getType().cast<FC::RefType>();
  assert(dstType.getEleTy().isa<FC::ArrayType>());
  assert(src.getType().isa<FC::ArrayType>());

  return builder.create<FC::FCStoreOp>(dst.getLoc(), src, dst);
}

FC::FCFuncOp
CodeGen::getMLIRFuncOpFor(Symbol *calledSymbol, ProgramUnit *calledPU,
                          llvm::SmallVectorImpl<mlir::Value> &argList) {

  auto name = cgHelper->getFunctionNameForSymbol(calledSymbol);
  FC::FCFuncOp mlirFunc;
  auto mlirloc = getLoc(calledSymbol->getSourceLoc());

  auto symref = getSymbolScopeList(calledSymbol);
  auto op = getOpForSymRef(symref);
  if (auto funcOp = llvm::dyn_cast_or_null<FC::FCFuncOp>(op)) {
    return funcOp;
  }

  llvm::SmallVector<mlir::Type, 2> argTys;
  for (auto arg : argList) {
    argTys.push_back(arg.getType());
  }
  auto funcType = cgHelper->getMLIRTypeFor(calledSymbol->getType())
                      .cast<mlir::FunctionType>();
  auto mlirFuncType = mlir::FunctionType::get(argTys, funcType.getResults(),
                                              theModule->getContext());

  mlirFunc = FC::FCFuncOp::create(mlirloc, name, mlirFuncType);

  if (calledPU && calledPU->isNestedUnit()) {
    context.currFn.addNestedFunction(mlirFunc);
  } else if (symref.getRootReference() == symref.getLeafReference()) {
    theModule->push_back(mlirFunc);
  }
  return mlirFunc;
}

mlir::Operation *CodeGen::emitCall(Symbol *symbol, ExprList &argsList,
                                   bool isSubroutineCall) {
  FC::FCCallOp op;
  if (symbol->getName() == "trim") {
    return emitTrimCall(symbol, argsList);
  }
  if (symbol->getName() == "memcpy") {
    return handleMemCopyCall(symbol, argsList);
  }

  if (symbol->getName() == "get_command_argument") {
    return handleCmdLineArgs(symbol, argsList);
  }

  auto fcFuncTy = llvm::dyn_cast<fc::FunctionType>(symbol->getType());
  if (!fcFuncTy) {
    llvm_unreachable("Called symbol is not a function.");
  }

  llvm::SmallVector<mlir::Value, 2> funcArgList;
  for (auto arg : llvm::enumerate(argsList)) {
    auto argVal = emitExpression(arg.value(), true);
    if (!argVal.getType().isa<FC::RefType>()) {
      if (argVal.getType().isIndex()) {
        // TODO: hard coded.
        argVal = emitCastExpr(argVal, builder.getIntegerType(32));
      }
      auto tempName = cgHelper->getTempUniqueName();
      auto memRefTy = FC::RefType::get(argVal.getType());
      auto alloc = createAlloca(memRefTy, tempName);
      auto loc = argVal.getLoc();
      auto storeOp = builder.create<FC::FCStoreOp>(loc, argVal, alloc);
      storeOp.setAttr("name", builder.getStringAttr(tempName));
      funcArgList.push_back(alloc);
      continue;
    }

    if (fcFuncTy->getArgType(0)->isVarArgTy()) {
      funcArgList.push_back(argVal);
      continue;
    }

    auto expectedType = fcFuncTy->getArgType(arg.index());
    auto currType = arg.value()->getType();
    if (expectedType != currType) {
      if (expectedType->isDynArrayTy() && currType->isArrayTy()) {
        auto expArr = cgHelper->getMLIRTypeFor(expectedType);
        auto ref = FC::RefType::get(expArr);
        argVal = emitCastExpr(argVal, ref);
      } else {
        assert(false && "wrong argument type for call statement!");
      }
    }
    funcArgList.push_back(argVal);
  }

  auto PU = cgHelper->getCalledProgramUnit(symbol);
  auto mlirloc = getLoc(symbol->getSourceLoc());

  auto mlirFunc = getMLIRFuncOpFor(symbol, PU, funcArgList);
  assert(mlirFunc);

  mlir::SymbolRefAttr attr = getSymbolScopeList(symbol);
  return builder.create<FC::FCCallOp>(
      mlirloc, attr, mlirFunc.getType().getResults(), funcArgList);
}

bool CodeGen::emitDeAllocateStmt(DeAllocateStmt *stmt) {
  SymbolList list = stmt->getDeAllocateObjList();
  auto stat = stmt->getStat();
  assert(!list.empty());
  auto loc = getLoc(stmt->getSourceLoc());

  for (unsigned i = 0; i < list.size(); ++i) {

    auto sym = list[i]->getOrigSymbol();
    auto arrayTy = llvm::dyn_cast<ArrayType>(sym->getType());
    assert(arrayTy);
    assert(arrayTy->isDynArrayTy());

    if (std == Standard::f77) {
      if (sym->getAllocKind() == fc::AllocationKind::StaticLocal) {
        context.functionAllocMap[context.symbolMap[sym->getName()]] = false;
      }
    }

    auto val = context.getMLIRValueFor(sym->getName());
    builder.create<FC::DeallocaOp>(loc, val);
  }

  // FIXME : Free doesn't return anything. Currently storing success value as
  // 0
  if (stat) {
    auto statValue = emitExpression(stat, true);
    auto zero = builder.create<mlir::ConstantIntOp>(loc, 0, 32);
    builder.create<FC::FCStoreOp>(loc, zero, statValue);
  }
  return true;
}

bool CodeGen::emitAllocateStmt(AllocateStmt *stmt) {
  SymbolList list = stmt->getAllocateObjList();
  assert(!list.empty());
  for (unsigned i = 0; i < list.size(); ++i) {
    auto arraySpec = stmt->getAllocateShape(i);
    auto loc = getLoc(stmt->getSourceLoc());
    unsigned numBounds = arraySpec->getNumBounds();
    llvm::SmallVector<mlir::Value, 2> operands;

    // Collect dynamic lower and upper bounds.
    for (unsigned j = 0; j < numBounds; ++j) {
      auto fcBounds = arraySpec->getBounds(j);
      auto lower = castToIndex(emitExpression(fcBounds.first));
      operands.push_back(lower);
      auto upper = castToIndex(emitExpression(fcBounds.second));
      operands.push_back(upper);
    }

    auto arrType = cgHelper->getMLIRTypeFor(list[i]->getType());
    auto arrAlloc = builder.create<FC::AllocaOp>(
        loc, list[i]->getName(), FC::RefType::get(arrType), operands);
    context.symbolMap[list[i]->getName()] = arrAlloc.getResult();
    auto allocation = arrAlloc.getResult();

    if (std == Standard::f77) {
      auto sym = list[i]->getOrigSymbol();
      if (sym->getAllocKind() == fc::AllocationKind::StaticLocal) {
        context.functionAllocMap[allocation] = true;
      }
    }
  }

  return true;
}

// TODO:Stop statement with string expression is not being supported yet
bool CodeGen::emitStopStmt(StopStmt *stmt) {
  llvm::SmallVector<mlir::Type, 2> argTys;
  llvm::SmallVector<mlir::Value, 2> funcArgList;
  auto funcName = "__fc_runtime_stop_int";
  mlir::Value exprVal = nullptr;
  auto mlirloc = getLoc(stmt->getSourceLoc());
  FC::FCFuncOp mlirFunc = nullptr;

  if (stmt->getStopCode() == nullptr) {
    exprVal = builder.create<mlir::ConstantIntOp>(mlirloc, 0, 64);
    argTys.push_back(exprVal.getType());
  } else {
    auto expr = llvm::dyn_cast<Expr>(stmt->getStopCode());
    exprVal = emitExpression(expr);
    argTys.push_back(exprVal.getType());
  }
  auto func = theModule->lookupSymbol(funcName);
  if (!func) {
    auto mlirFuncType =
        mlir::FunctionType::get(argTys, {}, theModule->getContext());

    mlirFunc = FC::FCFuncOp::create(mlirloc, funcName, mlirFuncType);
    calledFuncs[funcName] = mlirFuncType;
    theModule->push_back(mlirFunc);
  } else {
    mlirFunc = llvm::dyn_cast<FC::FCFuncOp>(func);
  }
  funcArgList.push_back(exprVal);
  builder.create<FC::FCCallOp>(getLoc(stmt->getSourceLoc()), mlirFunc,
                               funcArgList);
  return true;
}

bool CodeGen::emitExitStmt(ExitStmt *stmt) {
  mlir::Block *exitBB;
  auto mlirloc = getLoc(stmt->getSourceLoc());

  if (stmt->hasConstructName()) {
    assert(context.nameLoopMap.find(stmt->getConstructName()) !=
           context.nameLoopMap.end());
    auto cgLoop = context.nameLoopMap[stmt->getConstructName()];
    exitBB = cgLoop->getExitBB();
  } else {
    assert(context.currLoopVector.size() > 0);
    exitBB = context.currLoopVector.back()->getExitBB();
  }

  auto nextBB = context.currFn.addBlock();
  SmallVector<mlir::Value, 2> e1, e2;
  auto trueVal = builder.create<mlir::ConstantIntOp>(mlirloc, 1, 1);
  builder.create<mlir::CondBranchOp>(mlirloc, trueVal, exitBB, e1, nextBB, e2);

  context.currBB = nextBB;
  builder.setInsertionPointToEnd(nextBB);
  return true;
}

bool CodeGen::emitCycleStmt(CycleStmt *stmt) {
  mlir::Block *latchBB;
  auto mlirloc = getLoc(stmt->getSourceLoc());

  if (stmt->hasConstructName()) {
    assert(context.nameLoopMap.find(stmt->getConstructName()) !=
           context.nameLoopMap.end());
    auto cgLoop = context.nameLoopMap[stmt->getConstructName()];
    latchBB = cgLoop->getLatchBB();
  } else {
    assert(context.currLoopVector.size() > 0);
    latchBB = context.currLoopVector.back()->getLatchBB();
  }

  auto nextBB = context.currFn.addBlock();
  SmallVector<mlir::Value, 2> e1, e2;
  auto trueVal = builder.create<mlir::ConstantIntOp>(mlirloc, 1, 1);
  builder.create<mlir::CondBranchOp>(getLoc(stmt->getSourceLoc()), trueVal,
                                     latchBB, e1, nextBB, e2);

  context.currBB = nextBB;
  builder.setInsertionPointToEnd(nextBB);
  return true;
}

bool CodeGen::emitExectubaleConstructList(StmtList &stmtList) {
  if (stmtList.empty())
    return true;

  for (auto actionStmt : stmtList) {
    if (isa<Expr>(actionStmt)) {
      continue;
    }

    setCurrLineForDebug(actionStmt->getSourceLoc());

    switch (actionStmt->getStmtType()) {
    case AssignmentStmtKind:
      emitAssignment(static_cast<AssignmentStmt *>(actionStmt));
      break;
    case PrintStmtKind:
      emitPrintStmt(static_cast<PrintStmt *>(actionStmt));
      break;
    case CloseStmtKind:
      emitCloseStmt(static_cast<CloseStmt *>(actionStmt));
      break;
    case IfElseStmtKind:
      emitIfElseStmt(static_cast<IfElseStmt *>(actionStmt));
      break;
    case DoWhileStmtKind:
      emitFCDoWhileLoop(static_cast<DoWhileStmt *>(actionStmt));
      break;
    case DoStmtKind:
      emitDoStmt(static_cast<DoStmt *>(actionStmt));
      break;
    case CallStmtKind:
      emitCallStmt(static_cast<CallStmt *>(actionStmt));
      break;
    case WriteStmtKind:
      emitWriteStmt(static_cast<WriteStmt *>(actionStmt));
      break;
    case ReadStmtKind:
      emitReadStmt(static_cast<ReadStmt *>(actionStmt));
      break;
    case OpenStmtKind:
      emitOpenStmt(static_cast<OpenStmt *>(actionStmt));
      break;
    case ReturnStmtKind: {
      auto returnStmt = static_cast<ReturnStmt *>(actionStmt);
      auto exitVal = returnStmt->getExpr();
      assert(!exitVal);
      auto currBlock = builder.getBlock();
      auto hasSingleRegionParent =
          (llvm::isa<FC::DoOp>(currBlock->getParentOp()) ||
           llvm::isa<loop::IfOp>(currBlock->getParentOp()));
      if (!hasSingleRegionParent) {
        builder.create<FC::FCReturnOp>(getLoc(returnStmt->getSourceLoc()));
        auto deadBlock = builder.createBlock(builder.getBlock()->getParent());
        builder.setInsertionPointToStart(deadBlock);
      } else {
        builder.create<FC::DoReturnOp>(getLoc(returnStmt->getSourceLoc()));
      }
      break;
    }
    case AllocateStmtKind: {
      emitAllocateStmt(static_cast<AllocateStmt *>(actionStmt));
      break;
    }
    case StopStmtKind: {
      emitStopStmt(static_cast<StopStmt *>(actionStmt));
      break;
    }
    case ExitStmtKind: {
      emitExitStmt(static_cast<ExitStmt *>(actionStmt));
      break;
    }
    case CycleStmtKind: {
      emitCycleStmt(static_cast<CycleStmt *>(actionStmt));
      break;
    }
    case DeAllocateStmtKind: {
      emitDeAllocateStmt(static_cast<DeAllocateStmt *>(actionStmt));
      break;
    }
    case OpenMPParallelStmtKind: {
      emitOpenMPParallelStmt(static_cast<OpenMPParallelStmt *>(actionStmt));
      break;
    }
    default:
      llvm::errs() << "\n Unhandled statement: "
                   << actionStmt->dump(llvm::errs()) << "\n";
      llvm_unreachable("Unhandled Stmt kind");
    };
  }
  return true;
}

bool CodeGen::emitExecutionPart(ExecutionPart *execPart) {
  if (!execPart)
    return true;

  return emitExectubaleConstructList(execPart->getBlock()->getStmtList());
}