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
#include "codegen/CGASTHelper.h"

#include "AST/ProgramUnit.h"
#include "AST/SymbolTable.h"
#include "AST/Type.h"
#include "common/Debug.h"

#include "dialect/FCOps/FCOps.h"

using namespace fc;
using namespace ast;

CGASTHelper::CGASTHelper(ast::ParseTree *tree, mlir::OwningModuleRef &module,
                         mlir::OpBuilder &builder, Standard std)
    : parseTree(tree), TheModule(module), C(parseTree->getContext()),
      LLC(module->getContext()), builder(builder), std(std) {}

CGASTHelper::SubPUHelper *CGASTHelper::getSubPUHelper(ProgramUnit *PU) {
  auto val = subPUHelperMap.find(PU);
  if (val == subPUHelperMap.end())
    return nullptr;
  return &val->second;
}

mlir::Type CGASTHelper::getMLIRTypeFor(fc::Type *type) {
  switch (type->getTypeID()) {
  case fc::Type::LogicalID:
    return mlir::IntegerType::get(1, LLC);
  case fc::Type::CharacterID:
  case fc::Type::StringCharID:
    return mlir::IntegerType::get(8, LLC);
  case fc::Type::Int16ID:
    return mlir::IntegerType::get(16, LLC);
  case fc::Type::Int32ID:
    return mlir::IntegerType::get(32, LLC);
  case fc::Type::Int64ID:
    return mlir::IntegerType::get(64, LLC);
  case fc::Type::Int128ID:
    return mlir::IntegerType::get(32, LLC);
  case fc::Type::RealID:
    return mlir::FloatType::getF32(LLC);
  case fc::Type::DoubleID:
    return mlir::FloatType::getF64(LLC);
  case fc::Type::DummyArgID:
  case fc::Type::UndeclaredID:
  case fc::Type::UndeclaredFnID:
  case fc::Type::VarArgID:
    llvm_unreachable("Found unhandled type in codegen");

  case fc::Type::PointerID: {
    llvm_unreachable("Pointer type not handled");
  }
  // Array type:
  case fc::Type::ArrayID: {
    auto arrTy = static_cast<fc::ArrayType *>(type);
    auto baseTy = getMLIRTypeFor(arrTy->getElementTy());

    if (arrTy->isDynArrayTy()) {
      auto numBounds = arrTy->getNumDims();
      FC::ArrayType::Shape shape(numBounds);
      return FC::ArrayType::get(shape, baseTy);
    }
    auto dims = arrTy->getBoundsList();
    FC::ArrayType::Shape shape;
    for (auto &dim : dims) {
      shape.push_back({dim.first, dim.second, dim.second - dim.first + 1});
    }

    // TODO this is a hack, remove after string arrays are fixed in sema
    if (arrTy->isStringArrTy() && shape.size() == 2)
      std::reverse(shape.begin(), shape.end());

    return FC::ArrayType::get(shape, baseTy);
  }
  case fc::Type::FunctionID: {
    llvm::SmallVector<mlir::Type, 2> Tys;
    auto fcFuncTy = static_cast<FunctionType *>(type);
    bool varArgSeen = false;
    for (auto fcType : fcFuncTy->getArgList()) {
      if (fcType->isVarArgTy()) {
        // Should be the last argument.
        varArgSeen = true;
        break;
      }
      auto LLTy = getMLIRTypeFor(fcType);
      assert(LLTy);
      if (LLTy.isa<FC::RefType>())
        Tys.push_back(LLTy);
      else
        Tys.push_back(FC::RefType::get(LLTy));
    }

    if (fcFuncTy->getReturnType()->isVoidTy()) {
      return mlir::FunctionType::get(Tys, {}, LLC);
    }
    auto retTy = getMLIRTypeFor(fcFuncTy->getReturnType());
    return mlir::FunctionType::get(Tys, {retTy}, LLC);
  }

  case fc::Type::StructID: {
    llvm_unreachable("struct type not handled");
  }
  default:
    llvm_unreachable("type not handled yet.");
  };
}

mlir::Type CGASTHelper::getMLIRTypeFor(Symbol *symbol, bool returnMemRef) {
  auto parenSym = symbol->getParentSymbol();
  if (parenSym)
    symbol = parenSym;

  fc::Type *symType = symbol->getType(); // This will be OrigType

  mlir::Type MLTy;
  if (auto symPtrType = llvm::dyn_cast<fc::PointerType>(symType)) {
    if (symPtrType->getElementType()->isArrayTy())
      assert(symPtrType->getElementType()->isDynArrayTy());

    MLTy = getMLIRTypeFor(symPtrType);
  } else {
    MLTy = getMLIRTypeFor(symType);
  }

  if (!returnMemRef)
    return MLTy;
  if (MLTy.isa<FC::RefType>())
    return MLTy;

  return FC::RefType::get(MLTy);
}

void CGASTHelper::createSubPUHelper(ProgramUnit *PU) {
  auto parent = PU->getParent();
  if (!parent)
    return;

  if (parent->getKind() == ProgramUnitKind::ModuleKind)
    return;

  auto helperVal = subPUHelperMap.find(PU->getParent());
  if (helperVal == subPUHelperMap.end()) {

    // This is a nested subroutine. Get the used parent symbol list.
    SymbolSet set;
    // TODO: optimize.
    PU->getParent()->getUsedSymbolsInChildren(set);

    SubPUHelper helper;
    helper.hasFrameArg = true;
    helper.set = set;

    subPUHelperMap[PU->getParent()] = helper;
  }
}

// TODO: Remove
mlir::Type CGASTHelper::getMLIRTypeFor(ProgramUnit *PU, bool &hasFrameArg) {
  return getMLIRTypeFor(PU->getType());
}

std::string CGASTHelper::getEmittedNameForPU(std::string name) { return name; }

std::string CGASTHelper::getNameForProgramUnit(ProgramUnit *PU) {
  return PU->getName();
}

std::string CGASTHelper::getFunctionNameForSymbol(Symbol *sym) {
  return sym->getName();
}

// TODO: Remove.. unused...
std::string CGASTHelper::getGlobalSymbolName(Symbol *symbol) {

  if (symbol->getSymTable()->isModuleScope()) {
    return "_FCMOD_" + symbol->getOriginalModName() + "_" +
           symbol->getName().str();
  }

  auto name = symbol->getSymTable()->getName();
  std::string varName = name;
  if (!symbol->getSymTable()->isMainProgramScope())
    varName = getEmittedNameForPU(name);

  return varName + "_" + symbol->getName().str();
}

ProgramUnit *CGASTHelper::getCalledProgramUnit(Symbol *symbol) {

  symbol = symbol->getOrigSymbol();
  assert(symbol->getType()->isFunctionTy());

  auto symTable = symbol->getSymTable();
  if (symTable->isLoadedFromModFile())
    return nullptr;

  auto PU = symTable->getProgramUnit();
  assert(PU);

  // now search for the symbol in program unit and return.
  for (auto subPU : PU->getProgramUnitList()) {
    if (subPU->getName() == symbol->getName())
      return subPU;
  }
  return nullptr;
}

mlir::Value CGASTHelper::getReturnValueFor(std::string func) {
  assert(FuncRetMap.find(func) != FuncRetMap.end());
  return FuncRetMap[func];
}

unsigned CGASTHelper::getSizeForType(fc::Type *Ty) {
  switch (Ty->getTypeID()) {
  case fc::Type::Int32ID:
    return 4;
  case fc::Type::Int64ID:
    return 8;
  case fc::Type::RealID:
    return 4;
  case fc::Type::DoubleID:
    return 8;
  case fc::Type::CharacterID:
  case fc::Type::StringCharID:
    return 1;
  default:
    llvm_unreachable("getSizeForType(): Unhandled type");
  }
  return 0;
}