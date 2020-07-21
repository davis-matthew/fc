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
#ifndef FC_MLIR_CG_AST_HELPER_H
#define FC_MLIR_CG_AST_HELPER_H

#include "AST/ParserTreeCommon.h"
#include "common/Source.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

#include <map>

namespace fc {

class ArrayType;

using namespace ast;

// Helper calls for conversion from AST types/ names to LLVM IR.
class CGASTHelper {
private:
  ast::ParseTree *parseTree;
  mlir::OwningModuleRef &TheModule;
  ASTContext &C;
  mlir::MLIRContext *LLC;
  mlir::OpBuilder &builder;
  // Map to track functions and their return global value
  std::map<std::string, mlir::Value> FuncRetMap;

  // Map to keep track of original PU name emitted PU name
  std::map<std::string, std::string> PUNameMap;

  // This structure is to help emission of
  // nested subroutines.
  struct SubPUHelper {
    // Whether there is any frame arg.
    bool hasFrameArg;
    // What are the symbols to be passes to the
    // nested routines.
    SymbolSet set;
    // Starting index for extra arguments which are symbols
    // in parent PU used in this child PU.
    unsigned startIndex;

    SubPUHelper() : hasFrameArg(false) {}

    SubPUHelper(const SubPUHelper &other) {
      this->hasFrameArg = other.hasFrameArg;
      this->set = other.set;
      this->startIndex = other.startIndex;
    }
  };

  std::map<ProgramUnit *, SubPUHelper> subPUHelperMap;

  // Fortran standard
  Standard std;

public:
  explicit CGASTHelper(ast::ParseTree *tree, mlir::OwningModuleRef &module,
                       mlir::OpBuilder &builder, Standard std);
  mlir::Type getMLIRTypeFor(fc::Type *type);

  mlir::Type getMLIRTypeFor(Symbol *symbol, bool returnMemRef = false);

  mlir::Type getMLIRTypeFor(ProgramUnit *PU, bool &hasFrameArg);

  std::string getNameForProgramUnit(ProgramUnit *PU);

  bool emitDeclarationFor(ProgramUnit *sub) { return true; }

  std::string getFunctionNameForSymbol(Symbol *symbol);

  std::string getGlobalSymbolName(Symbol *symbol);

  SymbolList getUsedSymbolsInChildren();

  SubPUHelper *getSubPUHelper(ProgramUnit *PU);

  ProgramUnit *getCalledProgramUnit(Symbol *symbol);

  std::string getEmittedNameForPU(std::string name);

  mlir::Value getReturnValueFor(std::string fun);

  void createSubPUHelper(ProgramUnit *PU);

  unsigned getSizeForType(fc::Type *Ty);

  std::string getTempUniqueName() {
    static int count = 0;
    return "cg_temp." + std::to_string(count++);
  }
};
} // namespace fc

#endif
