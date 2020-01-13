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

#include "codegen/CodeGen.h"
#include "AST/Declaration.h"
#include "AST/ParserTreeCommon.h"
#include "AST/ProgramUnit.h"
#include "AST/Stmt.h"
#include "AST/SymbolTable.h"
#include "AST/Type.h"
#include "codegen/CGASTHelper.h"
#include "common/Debug.h"
#include "dialect/FCOps/FCOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"

#include "llvm-c/Target.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

using namespace fc;
using namespace ast;
using namespace llvm;

static llvm::cl::opt<bool>
    DumpBeforeVerify("dump-before-verify",
                     llvm::cl::desc("Dump MLIR module before verification"),
                     llvm::cl::init(false));

CodeGen::CodeGen(ASTContext &C, mlir::OwningModuleRef &theModule,
                 mlir::MLIRContext &mlirContext, Standard std)
    : ASTProgramPass(C, "LLVM CodeGenerator Pass"), C(C),
      mlirContext(mlirContext), builder(&mlirContext), theModule(theModule),
      std(std) {}

FC::FCLoadOp CodeGen::emitLoadInstruction(mlir::Value V,
                                          const llvm::Twine &Name,
                                          bool disableTBAA) {

  auto loadOp = builder.create<FC::FCLoadOp>(builder.getUnknownLoc(), V);
  loadOp.setAttr("name", builder.getStringAttr(Name.getSingleStringRef()));
  return loadOp;
}

void CodeGen::emitStoreInstruction(mlir::Value V, mlir::Value Ptr,
                                   bool disableTBAA) {
  builder.create<FC::FCStoreOp>(builder.getUnknownLoc(), V, Ptr);
}

void CodeGen::setCurrLineForDebug(SourceLoc loc) {}

mlir::Block *CodeGen::getNewBlock(mlir::Block *insertBefore) {
  auto insertPt = builder.saveInsertionPoint();
  mlir::Region::iterator it(insertBefore);
  mlir::Block *block;
  if (context.currRegion)
    block = builder.createBlock(context.currRegion, it);
  else
    block = builder.createBlock(insertBefore);
  builder.restoreInsertionPoint(insertPt);
  return block;
}

mlir::Value CodeGen::getValue(Symbol *symbol) {
  auto parenSym = symbol->getParentSymbol();
  if (parenSym)
    symbol = parenSym;

  return context.symbolMap[symbol->getName()];
}

FC::AllocaOp CodeGen::createAlloca(mlir::Type type, llvm::StringRef name) {
  llvm::SmallVector<mlir::Value, 2> vals;
  auto alloc = builder.create<FC::AllocaOp>(builder.getUnknownLoc(), name,
                                            type.cast<FC::RefType>(), vals);
  return alloc;
}

FC::AllocaOp CodeGen::createAlloca(Symbol *symbol) {
  auto llType = cgHelper->getMLIRTypeFor(symbol);
  if (!llType.isa<FC::RefType>()) {
    llType = FC::RefType::get(llType);
  }
  auto alloc = builder.create<FC::AllocaOp>(getLoc(symbol->getSourceLoc()),
                                            symbol->getName(),
                                            llType.cast<FC::RefType>());
  return alloc;
}

mlir::Value CodeGen::getIntentArgForSymbol(Symbol *symbol) {
  unsigned argNum = 0;
  assert(context.currPU->isSubroutine() || context.currPU->isFunction());
  auto sub = static_cast<Function *>(context.currPU);

  auto argsList = sub->getArgsList();
  for (auto arg : argsList) {
    if (arg.compare_lower(symbol->getName()) == 0)
      break;
    argNum++;
  }

  bool hasIntentArg = false;
  if (context.currPU->isNestedUnit()) {
    auto subPUHelper = cgHelper->getSubPUHelper(context.currPU->getParent());
    if (subPUHelper && subPUHelper->hasFrameArg) {
      hasIntentArg = true;
    }
  }

  if (hasIntentArg) {
    auto subPUHelper = cgHelper->getSubPUHelper(context.currPU->getParent());
    if (argNum >= subPUHelper->startIndex) {
      llvm_unreachable("Something went wrong with frame args");
    }
  }

  unsigned I = 0;
  for (auto LLArg = context.currFn.args_begin();
       LLArg != context.currFn.args_end(); ++LLArg) {
    if (I == argNum) {
      return (*LLArg);
    }
    I++;
  }
  return nullptr;
}

bool CodeGen::updatesymbolMapForIntentArg(Symbol *symbol) {
  auto Arg = getIntentArgForSymbol(symbol);
  assert(Arg);
  context.symbolMap[symbol->getName()] = Arg;
  return true;
}

void CodeGen::updateSymbolMapForFuncArg(Symbol *symbol) {
  assert(context.currPU->isFunction());
  auto func = static_cast<Function *>(context.currPU);

  unsigned argNum = 0;
  auto argsList = func->getArgsList();
  for (auto arg : argsList) {
    if (arg.compare_lower(symbol->getName()) == 0)
      break;
    argNum++;
  }

  unsigned i = 0;
  mlir::Value Arg;
  for (auto LLArg = context.currFn.args_begin();
       LLArg != context.currFn.args_end(); ++LLArg) {
    if (i == argNum) {
      Arg = (*LLArg);
      break;
    }
    ++i;
  }

  context.symbolMap[symbol->getName()] = Arg;
}

// This function handles the initializer for all the local variables.
bool CodeGen::emitEntityDecl(EntityDecl *entityDecl) {
  auto sym = entityDecl->getSymbol();
  auto LLVar = context.getMLIRValueFor(sym->getName());
  assert(LLVar && "Could not find MLIR value allocated!");
  auto Init = entityDecl->getInit();
  mlir::Value InitExpr = nullptr;
  fc::Type *Ty = sym->getType();

  if (sym->getAllocKind() == AllocationKind::Argument) {
    return true;
  }

  if (sym->getAllocKind() == AllocationKind::StaticLocal) {
    if (Init) {
      if (auto Const = llvm::dyn_cast<ConstantVal>(Init)) {
        auto type = Const->getType();
        if (type->isArrayTy()) {
          llvm::SmallVector<llvm::StringRef, 2> constList;
          for (auto &val : Const->getConstant()->getArrValue()) {
            constList.push_back(val);
          }
          InitExpr = emitConstant(constList, Const->getType(), Ty, true);
        } else {
          InitExpr = emitExpression(Init, false);
        }
      } else {
        InitExpr = emitExpression(Init, false);
      }
      emitStoreInstruction(InitExpr, LLVar);
    }
    return true;
  }

  if (sym->getAllocKind() == AllocationKind::StaticGlobal) {
    if (Init) {
      auto InitConst = dyn_cast<ConstantVal>(Init);
      assert(InitConst);
      FC::AllocaOp global;
      global = llvm::dyn_cast<FC::AllocaOp>(LLVar.getDefiningOp());
      if (!global) {
        return true;
      }
      auto eleType = global.getType().getEleTy();
      if (!global.getType().isStatic())
        return true;

      if (eleType.isa<FC::ArrayType>() &&
          global.getType().getUnderlyingEleType().isF32()) {
        llvm::SmallVector<float, 2> constList;
        for (auto &val : InitConst->getConstant()->getArrValue()) {
          constList.push_back(std::stof(val));
        }
        auto attr = builder.getF32ArrayAttr(constList);
        global.setAttr("value", attr);
        return true;
      }
      if (sym->getType()->isStringArrTy()) {
        auto arrTy = static_cast<ArrayType *>(sym->getType());
        assert(!arrTy->isDynArrayTy());
        assert(arrTy->getNumDims() == 1);
        auto bound = arrTy->getBoundsList()[0];
        auto size = bound.second - bound.first + 1;
        auto initRef = InitConst->getValue();

        // Padding the string with \0 to match size
        for (int i = initRef.size(); i < size; ++i)
          initRef.push_back('\0');

        global.setAttr("value", mlir::StringAttr::get(initRef, &mlirContext));
      } else if (eleType.isa<mlir::IntegerType>()) {

        int64_t value;
        if (eleType.isInteger(1)) {
          value = InitConst->getBool();
        } else {
          value = InitConst->getInt();
        }
        global.setAttr("value", mlir::IntegerAttr::get(eleType, value));

      } else if (eleType.isa<mlir::FloatType>()) {
        double d;
        if (InitConst->getType()->isFloatingTy()) {
          d = InitConst->getFloat();
        } else {
          int value;
          if (InitConst->getValueRef().consumeInteger(10, value))
            assert(false);
          d = value;
        }
        global.setAttr("value", mlir::FloatAttr::get(eleType, d));
      } else {
        assert("unknown type");
      }
    }
    return true;
  }
  llvm_unreachable("unhandled Allocation type");
}

bool CodeGen::createGlobalExternForSymbol(Symbol *symbol) { return true; }

bool CodeGen::updateArgForNestProgramUnit() { return true; }

void CodeGen::emitParentVariableAccess(Symbol *sym) {
  auto symRef = getSymbolScopeList(sym);
  auto op = getOpForSymRef(symRef);

  mlir::Type modVarType;
  auto moduleVar = llvm::dyn_cast_or_null<FC::AllocaOp>(op);
  if (!moduleVar) {
    auto captureOp = llvm::dyn_cast_or_null<FC::CaptureArgOp>(op);
    if (!captureOp) {
      auto PU = sym->getOrigSymbol()->getSymTable()->getProgramUnit();
      assert(PU);
      auto func = llvm::cast<fc::ast::Function>(PU);
      auto argNum = func->getArgNumForSymbol(sym->getName());
      auto funcOp = context.currFn.getParentOfType<FC::FCFuncOp>();
      assert(funcOp);
      auto arg = funcOp.getArgument(argNum);
      auto insertpt = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(&funcOp.front());
      builder.create<FC::CaptureArgOp>(getLoc(sym->getSourceLoc()), arg,
                                       sym->getName());
      builder.restoreInsertionPoint(insertpt);
      modVarType = arg.getType();
    } else {
      modVarType = captureOp.getType();
    }
  } else {
    moduleVar.setCaptured(builder.getBoolAttr(true));
    modVarType = moduleVar.getType();
  }

  auto addrOfOp = builder.create<FC::GetElementRefOp>(
      getLoc(sym->getSourceLoc()), symRef, modVarType.cast<FC::RefType>());
  context.symbolMap[sym->getName()] = addrOfOp;
}

mlir::SymbolRefAttr CodeGen::getSymbolScopeList(Symbol *sym) {

  sym = sym->getOrigSymbol();
  auto symTable = sym->getSymTable();

  // Belongs to global scope.
  if (sym->getParentSymbol() == nullptr && sym->getType()->isFunctionTy()) {
    auto PU = symTable->getProgramUnit();
    if (PU && PU->getProgramUnitList().empty() && !symTable->isModuleScope()) {
      return builder.getSymbolRefAttr(sym->getName());
    }
  }
  llvm::SmallVector<FlatSymbolRefAttr, 2> attrList;
  while (symTable && !symTable->isGlobalScope()) {
    attrList.push_back(builder.getSymbolRefAttr(symTable->getName()));
    symTable = symTable->getParent();
  }
  // std::reverse(attrList.begin(), attrList.end());
  auto symRef = builder.getSymbolRefAttr(sym->getName(), attrList);

  return symRef;
}

mlir::Operation *CodeGen::getOpForSymRef(mlir::SymbolRefAttr symRef) {
  mlir::SymbolTable table(theModule.get());
  mlir::Operation *op;
  for (auto ref : llvm::reverse(symRef.getNestedReferences())) {
    op = table.lookup(ref.getValue());
    table = mlir::SymbolTable(op);
    assert(op);
  }
  op = table.lookup(symRef.getRootReference());
  return op;
}

void CodeGen::emitModuleVariableAccess(Symbol *sym) {
  auto symRef = getSymbolScopeList(sym);
  auto op = getOpForSymRef(symRef);
  assert(op);
  auto moduleVar = llvm::dyn_cast<FC::AllocaOp>(op);
  assert(moduleVar);
  auto addrOfOp = builder.create<FC::GetElementRefOp>(
      getLoc(sym->getSourceLoc()), symRef, moduleVar.getType());
  context.symbolMap[sym->getName()] = addrOfOp;
}

bool CodeGen::emitSpecificationPart(SpecificationPart *specPart) {
  // TODO May be not necessary
  bool isFunction = context.currPU->isFunction();

  // Process remaining symbols which needs memory allocation.
  auto currSymTable = context.currPU->getSymbolTable();
  for (auto sym : currSymTable->getSymbolList()) {

    setCurrLineForDebug(sym->getSourceLoc());

    // Already allocated..
    if (context.getMLIRValueFor(sym->getName()) != nullptr)
      continue;

    // Point to the right argument.
    if (sym->hasIntent()) {
      updatesymbolMapForIntentArg(sym);
      continue;
    }

    if (sym->getAllocKind() == StaticLocal) {

      // If it is a dynamic array type in f77. Ignore .
      // Will be malloc'ed later.
      if (sym->getType()->isDynArrayTy() && sym->getParentSymbol() == nullptr) {
        continue;
      }
      auto Alloca = createAlloca(sym);
      /* auto Alloca = createAllocStatic(sym); */
      context.symbolMap[sym->getName()] = Alloca.getResult();
      continue;
    }

    // Function types needs no allocation here.
    if (sym->getType()->isFunctionTy())
      continue;

    if (sym->getType()->isUndeclared()) {
      auto parentSym = sym->getParentSymbol();
      if (sym->getType()->isUndeclaredFnTy() && !parentSym) {
        continue;
      }
      assert(parentSym);
      if (parentSym->getType()->isFunctionTy()) {
        continue;
      }
    }

    // These are mostly extern global symbols.
    if (sym->getAllocKind() == StaticGlobal) {
      auto Alloca = createAlloca(sym);
      /* auto Alloca = createAllocStatic(sym); */
      Alloca.setAttr("alloc_kind", builder.getStringAttr("static"));
      context.symbolMap[sym->getName()] = Alloca.getResult();
      continue;
    }

    if (sym->getAllocKind() == Argument && isFunction) {
      updateSymbolMapForFuncArg(sym);
      continue;
    }

    // If these are local symbols in the parent
    // routine, ignore it for now. Handled, below.
    auto parentSym = sym->getParentSymbol();
    assert(parentSym);
    sym = parentSym;

    if (sym->getSymTable()->isModuleScope()) {
      emitModuleVariableAccess(sym);
      continue;
    }

    if (sym->getAllocKind() == StaticGlobal) {
      if (!createGlobalExternForSymbol(sym)) {
        return false;
      }
      continue;
    }

    if (context.currPU->isNestedUnit()) {
      emitParentVariableAccess(parentSym);
      continue;
    }
    assert(context.currPU->isNestedUnit());
    assert(sym->getSymTable() == context.currPU->getParent()->getSymbolTable());
  }

  if (context.currPU->isNestedUnit()) {
    if (!updateArgForNestProgramUnit()) {
      return false;
    }
  }

  if (!specPart || !specPart->getBlock())
    return true;

  for (auto stmt : specPart->getBlock()->getStmtList()) {
    // DeclEliminatorPass should have removed entityDecls for program-units
    // that doesn't have global-vars (like module, subroutine/function with a
    // child etc.)
    if (auto entityDecl = llvm::dyn_cast<EntityDecl>(stmt)) {
      if (!emitEntityDecl(entityDecl)) {
        return false;
      }
    }
  }

  return true;
}

void CodeGen::emitDebugMetaForFunction() {}
bool CodeGen::createSubPUHelpers(ProgramUnit *PU) {
  for (auto subPU : PU->getProgramUnitList()) {
    cgHelper->createSubPUHelper(subPU);
    createSubPUHelpers(subPU);
  }
  return true;
}

bool CodeGen::emitFunctionDeclaration(ProgramUnit *PU) {
  if (llvm::isa<fc::Function>(PU)) {
    cgHelper->emitDeclarationFor(PU);
  }
  for (auto subPU : PU->getProgramUnitList()) {
    emitFunctionDeclaration(subPU);
  }
  return true;
}

bool CodeGen::emitASTModule(ast::Module *module) {
  auto loc = builder.getUnknownLoc();
  // TODO: loc
  auto fortranModuleOp = FC::FortranModuleOp::create(loc, module->getName());
  theModule->push_back(fortranModuleOp);
  builder.setInsertionPointToStart(&fortranModuleOp.body().front());
  emitSpecificationPart(module->getSpec());
  return true;
}

void CodeGen::deAllocateTemps() {
  for (auto II = context.functionAllocMap.begin();
       II != context.functionAllocMap.end(); ++II) {
    if (II->second) {
      builder.create<FC::DeallocaOp>(builder.getUnknownLoc(), II->first);
      II->second = false;
    }
  }
}

mlir::Location CodeGen::getLoc(fc::SourceLoc loc) {
  return builder.getFileLineColLoc(builder.getIdentifier(parseTree->getName()),
                                   loc.Line, loc.Col);
}

// Populate vector with extra args from parent
void CodeGen::populateExtraArgumentType(
    llvm::SmallVector<mlir::Type, 2> &captureList) {
  return;
}

FC::FCFuncOp CodeGen::emitFunction(fc::Function *func) {

  auto name = cgHelper->getNameForProgramUnit(func);
  bool hasFrameArg = false;
  auto funcType = cgHelper->getMLIRTypeFor(context.currPU, hasFrameArg)
                      .cast<mlir::FunctionType>();
  auto mlirloc = builder.getUnknownLoc();
  FC::FCFuncOp fn;
  if (func->isNestedUnit() || func->getParent()->isModule()) {
    switch (func->getParent()->getKind()) {
    case ModuleKind: {
      fn = FC::FCFuncOp::create(mlirloc, name, funcType);
      auto fortranModuleOp = theModule->lookupSymbol<FC::FortranModuleOp>(
          func->getParent()->getName());
      assert(fortranModuleOp);
      fortranModuleOp.push_back(fn);
    } break;
    default:
      fn = llvm::dyn_cast_or_null<FC::FCFuncOp>(
          context.currFn.lookupSymbol(name));
      if (!fn) {
        fn = FC::FCFuncOp::create(mlirloc, name, funcType);
        context.currFn.addNestedFunction(fn);
      }
      fn.setAttr("linkage_type", builder.getStringAttr("internal"));
      break;
    };
  } else {
    fn = llvm::dyn_cast_or_null<FC::FCFuncOp>(theModule->lookupSymbol(name));
    if (!fn) {
      fn = FC::FCFuncOp::create(mlirloc, name, funcType);
      theModule->push_back(fn);
    }
  }
  assert(fn);

  auto EntryBB = fn.addEntryBlock();
  context.currFn = fn;
  builder.setInsertionPointToEnd(EntryBB);
  context.currBB = EntryBB;

  emitSpecificationPart(func->getSpec());

  emitExecutionPart(func->getExecPart());

  mlirloc = builder.getUnknownLoc();

  if (func->isSubroutine()) {
    bool hasReturn = false;
    if (auto block = builder.getInsertionBlock()) {
      if (!block->empty()) {
        auto &op = block->back();
        if (op.isKnownTerminator()) {
          if (isa<FC::FCReturnOp>(op)) {
            hasReturn = true;
          }
        }
      }
    }
    if (!hasReturn)
      builder.create<FC::FCReturnOp>(mlirloc);
  } else if (func->isMainProgram()) {
    auto constant = builder.create<mlir::ConstantIntOp>(mlirloc, 0, 32);
    SmallVector<mlir::Value, 2> ops = {constant.getResult()};
    builder.create<FC::FCReturnOp>(mlirloc, ops);
  } else {
    auto returnVal = context.getMLIRValueFor(func->getName());
    assert(returnVal);
    auto type = returnVal.getType();

    // Hint the allocator to use the malloc for memory allocation.
    // TODO: It will be freed once the copy to the called function happens.
    // FIXME: Is there a better way to do this?
    if (auto allocOp =
            llvm::dyn_cast_or_null<FC::AllocaOp>(returnVal.getDefiningOp())) {
      auto refType = type.cast<FC::RefType>();
      if (refType.getEleTy().isa<FC::ArrayType>()) {
        allocOp.setAttr("use_malloc", builder.getBoolAttr(true));
      }
    }
    auto LI = builder.create<FC::FCLoadOp>(mlirloc, returnVal);
    SmallVector<mlir::Value, 2> ops = {LI};
    builder.create<FC::FCReturnOp>(mlirloc, ops);
  }

  builder.setInsertionPoint(context.currBB->getTerminator());
  deAllocateTemps();
  builder.setInsertionPointToEnd(context.currBB);
  return fn;
}

bool CodeGen::emitProgramUnit(ProgramUnit *PU) {

  llvm::StringRef fnName = "";
  FC::FCFuncOp Fn;
  auto kind = PU->getKind();
  context.currPU = PU;
  switch (kind) {
  case ProgramUnitKind::SubroutineKind:
  case ProgramUnitKind::FunctionKind: {
    auto function = static_cast<Function *>(PU);
    Fn = emitFunction(function);
    assert(Fn);
    fnName = Fn.getName();
    break;
  }

  case ProgramUnitKind::ModuleKind: {
    auto Mod = static_cast<Module *>(PU);
    fnName = Mod->getName();
    if (!emitASTModule(Mod)) {
      error() << "Error during module emission\n";
      return false;
    }
    break;
  }
  case ProgramUnitKind::MainProgramKind: {
    auto fcMain = static_cast<fc::Function *>(PU);
    fnName = fcMain->getName();
    Fn = emitFunction(fcMain);
    Fn.setAttr("main_program", builder.getBoolAttr(true));
    if (!Fn) {
      error() << "\n Error during MainProgram emission\n";
      return false;
    }
    break;
  }
  default:
    llvm_unreachable("unknown program unit found");
  };

  // Reset all the context info.
  context.reset();

  // Now emit all the nested subroutines.
  for (auto subPU : PU->getProgramUnitList()) {
    context.currPU = PU;
    context.currFn = Fn;
    if (!emitProgramUnit(subPU)) {
      error() << "\n Error during sub program emission\n";
      return false;
    }
  }

  return true;
}

bool CodeGen::runOnProgram(ParseTree *parseTree) {
  this->parseTree = parseTree;
  this->theModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(&mlirContext));

  debug() << C.inputFileName;
  debug() << this->std;
  // Set the codegen helper class.
  CGASTHelper cgHelper(parseTree, theModule, builder, std);
  this->cgHelper = &cgHelper;

  // TODO: Remove
  for (auto PU : parseTree->getProgramUnitList()) {
    if (!createSubPUHelpers(PU)) {
      error() << "MLIR CG: Failed to create mlir subou helpers\n";
      return false;
    }
  }

  // Emit all subroutine declarations.
  for (auto PU : parseTree->getProgramUnitList()) {
    if (!emitFunctionDeclaration(PU)) {
      error() << "LLVM CG: Failed to emit Subroutine declarations\n";
      return false;
    }
  }

  // Emit all Program units now.
  for (auto PU : parseTree->getProgramUnitList()) {
    if (!emitProgramUnit(PU)) {
      error() << "\n Error during external PU emission\n";
      return false;
    }
  }

  if (DumpBeforeVerify)
    theModule->dump();
  if (failed(mlir::verify(theModule.get()))) {
    theModule->emitError("module verification error");
    return false;
  }
  return true;
}
