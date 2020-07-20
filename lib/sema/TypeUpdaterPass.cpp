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
#include "AST/Declaration.h"
#include "AST/Expressions.h"
#include "AST/ProgramUnit.h"
#include "AST/SymbolTable.h"
#include "AST/Type.h"
#include "common/Debug.h"
#include "common/Diagnostics.h"

using namespace fc;
using namespace ast;
namespace fc {

class TypeUpdaterPass : public ASTPUPass {

private:
  ProgramUnit *currPU;
  bool onlyUpdateSpec{false};

  static long int getConstantValue(Expr *value, bool &found) {
    if (auto val = llvm::dyn_cast<ConstantVal>(value)) {
      found = true;
      return val->getInt();
    }
    if (auto objName = llvm::dyn_cast<ObjectName>(value)) {
      auto sym = objName->getSymbol();
      if (sym->getAttr().isConst) {
        auto constant = sym->getInitConstant();
        if (constant) {
          found = true;
          return constant->getInt();
        }
      }
    }
    found = false;
    return NULL;
  }

  void setCoreElementType(Type *&type, Type *eleType) {
    Type *baseType = type;
    PointerType *ptrType = nullptr;
    ArrayType *arrType = nullptr;

    if ((ptrType = llvm::dyn_cast<PointerType>(baseType))) {
      baseType = ptrType->getElementType();
    }

    if ((arrType = llvm::dyn_cast<ArrayType>(baseType))) {
      arrType->setElementTy(eleType);
      return;
    }

    if (ptrType) {
      ptrType->setElementTy(eleType);
      return;
    }

    type = eleType;
  }

  void updateDerivedTypeDef(DerivedTypeDef *dtd) {
    StructType *dtdType = dtd->getType();

    unsigned i = 0;

    // dtd's won't have a block if it's imported from .mod file.
    // TODO: with the addition of fc::NamedType, I think we can safely rely only
    // on dtd's symtab.
    if (dtd->getBlock()) {
      for (Stmt *stmt : dtd->getBlock()->getStmtList()) {
        if (auto entityDecl = llvm::dyn_cast<EntityDecl>(stmt)) {
          dtdType->setContainedType(entityDecl->getSymbol()->getType(), i++);
        }
      }
    } else {
      for (Symbol *sym : dtd->getSymbolTable()->getOrderedSymbolList()) {
        Type *symType = sym->getType();

        if (auto namedType =
                llvm::dyn_cast<NamedType>(Type::getCoreElementType(symType))) {
          DerivedTypeDef *namedTypeDTD = currPU->getDTD(namedType->getName());
          StructType *namedTypeAsStruct = namedTypeDTD->getType();
          assert(namedTypeAsStruct);

          setCoreElementType(symType, namedTypeAsStruct);
          dtdType->setContainedType(symType, i++);

        } else {
          dtdType->setContainedType(symType, i++);
          sym->setType(symType);
        }
      }
    }

    dtd->setCompleteType(true);
  }

  bool updateArrayType(Symbol *symbol, ArraySpec *spec) {
    auto arrTy = llvm::dyn_cast<ArrayType>(symbol->getType());
    if (!arrTy || !arrTy->getBoundsList().empty())
      return false;

    auto boundsList = arrTy->getBoundsList();

    if (spec->getBoundsList().empty() && boundsList.empty()) {
      symbol->setType(
          ArrayType::get(Context, arrTy->getElementTy(), spec->getNumDims()));
      return true;
    }

    for (auto dynBounds : spec->getBoundsList()) {
      ArrayBounds bounds;
      bool found;
      bounds.first = getConstantValue(dynBounds.first, found);
      if (!found) {
        return false;
      }
      bounds.second = getConstantValue(dynBounds.second, found);
      if (!found) {
        return false;
      }
      boundsList.push_back(bounds);
    }
    symbol->setType(ArrayType::get(Context, arrTy->getElementTy(), boundsList));
    return true;
  }

  void updateNamedTypes(ProgramUnit *PU) {
    SymbolTable *symTable = PU->getSymbolTable();
    for (Symbol *sym : symTable->getSymbolList()) {
      Type *symType = sym->getOrigSymbol()->getType();
      if (auto namedType =
              llvm::dyn_cast<NamedType>(Type::getCoreElementType(symType))) {
        DerivedTypeDef *namedTypeDTD = PU->getDTD(namedType->getName());
        StructType *namedTypeAsStruct = namedTypeDTD->getType();

        // Asserting because NamedTypes are generated by mod-file-dumper, so if
        // we come across a NamedType symbol, then we can be sure that this is a
        // post-module pass and we expect all our DTDs to be populated.
        assert(namedTypeAsStruct);
        setCoreElementType(symType, namedTypeAsStruct);

        // FIXME! See use-stmt-handler, part-ref resolver etc.
        sym->setType(symType);
        sym->getOrigSymbol()->setType(symType);

      } else if (auto funcType = llvm::dyn_cast<FunctionType>(symType)) {
        unsigned i = 0;
        for (Type *argType : funcType->getArgList()) {
          if (auto namedType = llvm::dyn_cast<NamedType>(
                  Type::getCoreElementType(argType))) {
            DerivedTypeDef *namedTypeDTD = PU->getDTD(namedType->getName());
            StructType *namedTypeAsStruct = namedTypeDTD->getType();

            setCoreElementType(argType, namedTypeAsStruct);
            funcType->setArgType(argType, i);
          }
          ++i;
        }
      }
    }
  }

  bool checkEntityDeclStmt(EntityDecl *entity) {
    // This function is a series of "pipelined phases" that "refines" the type
    // of entity. First we check if it's a derived-type, set type accordingly.
    // Then we consider it's attributes: eg.: If it has an array-spec, make it
    // an array of derived-type etc.
    //
    // Eg. type(foo), pointer :: x(10) set x as a derived-type first, then makes
    // it an array-of-derived-type, then finally it get's refined to
    // pointer-to-array-of-derived-type.

    DerivedTypeSpec *derivedTypeSpec = llvm::dyn_cast<DerivedTypeSpec>(
        entity->getDeclTypeSpec()->getTypeSpec());

    Symbol *sym = entity->getSymbol();

    // If entity is a type(derived-type) ...
    if (derivedTypeSpec) {
      DerivedTypeDef *dtd = currPU->getDTD(derivedTypeSpec->getName());

      // If this TypeUpdater is prior to the UseStmtHandler, then we won't have
      // the DTD in the currPU's spec-part.
      if (!dtd) {
        return true;
      }

      if (!dtd->isCompleteType()) {
        assert(sym->isPointer());
      }

      sym->setType(dtd->getType());
    }

    // Check if the array types are constants values and update the array
    // spec.
    auto arrSpec = entity->getArraySpec();
    if (arrSpec) {

      // If we have a dt-array, sym's type should be ArrayType(dtType)
      if (derivedTypeSpec && sym->getType()->isStructTy()) {
        StructType *dtType = llvm::dyn_cast<StructType>(sym->getType());
        assert(dtType);
        sym->setType(ArrayType::get(Context, dtType, arrSpec->getNumDims()));
      }
      updateArrayType(sym, arrSpec);
    }

    // If this pass is executed more than once, then we need to skip the
    // pointer-syms that got the pointer-type.
    if (sym->isPointer() && !sym->getType()->isPointerTy()) {
      sym->setType(PointerType::get(Context, sym->getType()));
    }

    return true;
  }

  bool updateEntityDeclStmts(StmtList &stmts) {
    for (auto stmt : stmts) {
      auto entityDeclStmt = llvm::dyn_cast<EntityDecl>(stmt);
      if (!entityDeclStmt)
        continue;
      if (!checkEntityDeclStmt(entityDeclStmt))
        return false;
    }
    return true;
  }

public:
  explicit TypeUpdaterPass(ASTContext &C, bool onlyUpdateSpec)
      : ASTPUPass(C, "TypeUpdaterPass"), onlyUpdateSpec(onlyUpdateSpec) {}

  virtual bool runOnProgramUnit(ProgramUnit *PU) override {

    currPU = PU;
    auto currSymTable = PU->getSymbolTable();
    auto specPart = PU->getSpec();

    if (specPart && specPart->getBlock()) {
      for (auto dtd : specPart->getDTDs()) {
        // This part makes types within derived-type-def concrete (ie. for eg.
        // arr[U, U] becomes arr[3, 4], derived-types within it point to the
        // corresponding dtd's type).
        // Note that dtd won't have a block if it's imported from a .mod file
        if (dtd->getBlock())
          updateEntityDeclStmts(dtd->getBlock()->getStmtList());

        // Make the dtd's StructType complete and concrete.
        updateDerivedTypeDef(dtd);
      }

      // 1. Update Array types in typedeclstmt.
      updateEntityDeclStmts(specPart->getBlock()->getStmtList());
    }

    updateNamedTypes(currPU);

    // If we're running this pass for solely updating the DTD after an import
    // from module. TODO: This is a quick-fix, we need to move this to a
    // separate sema pass.
    if (onlyUpdateSpec)
      return true;

    if (currSymTable->isSubroutineScope() || currSymTable->isFunctionScope()) {
      // 2. Update Subroutine arg types.
      // If subroutine arg is not declared in type declaration stmt,
      // assign default of inout with int32 type.

      ArgsList argsList;
      bool isFunction = false;
      Function *func = nullptr;
      if (PU->isFunction()) {
        func = static_cast<Function *>(PU);
        argsList = func->getArgsList();
        isFunction = true;
      } else {
        auto Sub = static_cast<Function *>(PU);
        argsList = Sub->getArgsList();
      }
      llvm::SmallVector<Type *, 2> argList;
      for (auto arg : argsList) {
        auto sym = currSymTable->getSymbol(arg);
        assert(sym);
        if (sym->getType()->isDummyArgTy()) {
          sym->setType(Type::getRealTy(Context));
          sym->setIntentKind(IntentKind::InOut);
          sym->setAllocKind(AllocationKind::Argument);
        }
        argList.push_back(sym->getType());
      }

      // 3. Now update the subroutine Function type in parent symbol table.
      auto parentSymTable = currSymTable->getParent();
      assert(parentSymTable);
      auto FuncSym = parentSymTable->getSymbol(PU->getName());
      assert(FuncSym && FuncSym->getType()->isFunctionTy());

      // For function either return type is explicitly specified or
      // an variable of the same name should be there in symbol table.
      if (isFunction) {
        assert(func);
        auto sym = currSymTable->getSymbol(PU->getName());
        assert(sym && "Function return value symbol not found");
        if (sym->getType()->isUndeclaredTy()) {
          auto funcType = llvm::dyn_cast<FunctionType>(FuncSym->getType());
          assert(funcType);
          auto funcRetType = funcType->getReturnType();
          assert(!funcRetType->isUndeclaredTy());
          sym->setType(funcRetType);
          sym->setAllocKind(AllocationKind::StaticLocal);
          func->setReturnType(funcRetType);
        } else {
          func->setReturnType(sym->getType());
        }
      }
      Type *retTy = Type::getVoidTy(Context);
      auto FuncTy = FunctionType::get(Context, retTy, argList);
      FuncSym->setType(FuncTy);
    }

    return true;
  }
}; // namespace fc

ASTPass *createTypeUpdaterPass(ASTContext &C, bool onlyUpdateSpec) {
  return new TypeUpdaterPass(C, onlyUpdateSpec);
}

} // namespace fc
