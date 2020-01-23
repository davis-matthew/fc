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
#ifndef FC_MLIR_CODEGEN_NEW_H_
#define FC_MLIR_CODEGEN_NEW_H_

#include "AST/ASTPass.h"
#include "AST/ParserTreeCommon.h"
#include "AST/Type.h"
#include "common/Source.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "dialect/FC/FCOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"

#include <map>
#include <memory>

namespace mlir {
class Module;
class FuncOp;
class Block;
class Location;
} // namespace mlir

namespace fc {

using namespace ast;

struct RuntimeHelper;
class CGASTHelper;

class CGLoop {

private:
  mlir::Block *header;
  mlir::Block *latch;
  mlir::Block *exit;
  llvm::StringRef name;

public:
  CGLoop(mlir::Block *_header, mlir::Block *_latch, mlir::Block *_exit,
         llvm::StringRef _name)
      : header(_header), latch(_latch), exit(_exit), name(_name) {}

  mlir::Block *getHeaderBB() {
    assert(header);
    return header;
  }

  mlir::Block *getLatchBB() {
    assert(latch);
    return latch;
  }

  mlir::Block *getExitBB() {
    assert(exit);
    return exit;
  }

  llvm::StringRef getName() { return name; }
};

class CodeGen : public ASTProgramPass {

public:
  struct CGContext {
    FC::FCFuncOp currFn;
    mlir::Block *currBB;
    fc::ast::ProgramUnit *currPU;
    mlir::Region *currRegion;
    llvm::StringMap<mlir::Value> symbolMap;
    // Map holds the file descripter and corresponding file unit number
    std::map<long int, mlir::Value> FDMap;

    // Map to track the loops
    // TODO May be redundant?
    std::map<Stmt *, CGLoop *> stmtLoopMap;
    std::map<llvm::StringRef, CGLoop *> nameLoopMap;
    mlir::SmallVector<CGLoop *, 2> currLoopVector;
    llvm::MapVector<mlir::Value, bool> functionAllocMap;

    bool needReturn;

    bool isParallelLoop;
    CGContext()
        : currFn(nullptr), currBB(nullptr), currPU(nullptr),
          currRegion(nullptr), needReturn(true), isParallelLoop(false) {}

  public:
    void reset() {
      currFn = nullptr;
      currBB = nullptr;
      currPU = nullptr;
      currRegion = nullptr;
      currLoopVector.clear();
      stmtLoopMap.clear();
      nameLoopMap.clear();
      FDMap.clear();
      symbolMap.clear();
      needReturn = true;
      isParallelLoop = false;
      functionAllocMap.clear();
    }

    mlir::Value getMLIRValueFor(llvm::StringRef ref) {
      auto val = symbolMap.find(ref.str());
      if (val == symbolMap.end())
        return nullptr;
      return val->second;
    }

    mlir::Value getFDFor(long int unit) { return FDMap[unit]; }
  };

  CodeGen(ASTContext &C, mlir::OwningModuleRef &theModule,
          mlir::MLIRContext &mlirContext, Standard std);

  void deAllocateTemps();

  bool emitProgramUnit(ProgramUnit *PU);

  void setCurrLineForDebug(SourceLoc loc);

  void emitDebugMetaForFunction();

  FC::FCFuncOp getMLIRFuncOpFor(Symbol *symbol, ProgramUnit *calledPU,
                                llvm::SmallVectorImpl<mlir::Value> &argList);

  FC::FCFuncOp emitFunction(Function *func);

  bool emitASTModule(ast::Module *mod);

  mlir::Value getValue(Symbol *symbol);

  std::string getTypeForProramUnit(ProgramUnit *PU);

  void populateExtraArgumentType(llvm::SmallVector<mlir::Type, 2> &argTypes);

  // Emit specification part.
  bool emitSpecificationPart(SpecificationPart *);

  void emitParentVariableAccess(Symbol *sym);

  void emitModuleVariableAccess(Symbol *sym);

  // Emit execution part.
  bool emitExecutionPart(ExecutionPart *execPart);

  bool emitExectubaleConstructList(StmtList &stmtList);

  // Emit EntityDecl
  bool emitEntityDecl(EntityDecl *entityDecl);

  bool createGlobalExternForSymbol(Symbol *sym);

  FC::FCFuncOp getOrInsertFuncOp(std::string name,
                                 llvm::ArrayRef<mlir::Type> argTys,
                                 mlir::Type retTy);

  /// \brief when \p constuctOnlyConstant is true, global constant value
  /// of type \p lhsTy is returned instead of GlobalVariable.
  mlir::Value emitConstant(mlir::ArrayRef<llvm::StringRef> valueList,
                           fc::Type *type, fc::Type *lhsTy = nullptr,
                           bool constuctOnlyConstant = false);

  mlir::Value emitSizeForArrBounds(ArrayBounds &);

  // Emit array-element represented by \p arrEle. Set \p addr if this array
  // belongs to some GEP'ed address (eg. in the case of array within a
  // struct-comp), ie. the mlir-arry for this arrEle is in \p addr.
  // This signature is followed by the analogous emitXXXArrayElement() APIs.
  mlir::Value emitArrayElement(ArrayElement *arrEle, bool isLHS = false,
                               mlir::Value addr = nullptr);

  mlir::Value emitFCArrayElement(ArrayElement *arrEle);

  mlir::Value emitStaticArrayElement(ArrayElement *expr, bool isLHS = false,
                                     mlir::Value addr = nullptr);

  mlir::Value emitDynArrayElement(ArrayElement *expr, bool isLHS = false,
                                  mlir::Value addr = nullptr);

  mlir::Value emitf77DynArrayElement(ArrayElement *expr, bool isLHS = false,
                                     mlir::Value addr = nullptr);

  mlir::Value emitExpression(Expr *expr, bool isLHS = false);

  mlir::Value emitCastExpr(mlir::Value from, mlir::Type toType);

  bool emitAssignment(AssignmentStmt *stmt);

  bool emitPointerAssignment(PointerAssignmentStmt *stmt);

  bool emitArraySectionStore(Expr *lhs, Expr *rhs, mlir::Location mlirloc);

  bool emitStopStmt(StopStmt *stmt);

  bool emitPrintStmt(PrintStmt *stmt);

  bool emitWriteStmt(WriteStmt *stmt);

  bool emitInternalWriteStmt(WriteStmt *stmt);

  bool emitReadStmt(ReadStmt *stmt);

  bool emitInternalReadStmt(ReadStmt *stmt);

  bool emitOpenStmt(OpenStmt *openStmt);

  bool emitCloseStmt(CloseStmt *closeStmt);

  bool emitIfElseStmt(IfElseStmt *stmt);

  bool emitLoopIfOperation(IfElseStmt *stmt);

  bool emitFCDoWhileLoop(DoWhileStmt *stmt);

  bool emitDoStmt(DoStmt *stmt);

  bool emitFCDoLoop(DoStmt *stmt);

  mlir::Value castToIndex(mlir::Value v);

  bool emitCallStmt(CallStmt *stmt);

  mlir::Value expandIntrinsic(FunctionReference *ref);

  mlir::Operation *emitTrimCall(Symbol *symbol, ExprList &argsList);

  mlir::Operation *handleMemCopyCall(Symbol *symbol, ExprList &argsList);

  mlir::Operation *handleCmdLineArgs(Symbol *symbol, ExprList &argsList);

  mlir::Operation *emitCall(Symbol *symbol, ExprList &exprList,
                            bool isSubroutineCall = false);

  bool emitMemCpy(CallStmt *stmt);

  bool emitCycleStmt(CycleStmt *stmt);

  bool emitDeAllocateStmt(DeAllocateStmt *stmt);

  bool emitAllocateStmt(AllocateStmt *stmt);

  bool emitExitStmt(ExitStmt *Stmt);

  bool updatesymbolMapForIntentArg(Symbol *);

  void updateSymbolMapForFuncArg(Symbol *);

  mlir::Value getArrDimSizeVal(Expr *expr, mlir::Value exprVal);

  mlir::Value castIntToFP(mlir::Value val, mlir::Type castToTy);

  mlir::Value getMLIRBinaryOp(mlir::Value lhsVal, mlir::Value rhsVal,
                              BinaryOpKind opKind);
  mlir::Value getMLIRComplexBinaryOp(mlir::Value lhsVal, mlir::Value rhsVal,
                                     BinaryOpKind opKind);
  mlir::Value getMLIRArrayBinaryOp(mlir::Value lhsVal, mlir::Value rhsVal,
                                   fc::ast::BinaryOpKind opKind);
  mlir::Value getMLIRRelationalOp(mlir::Value lhsVal, mlir::Value rhsVal,
                                  RelationalOpKind opKind);
  mlir::Value getMLIRArrayRelationalOp(mlir::Value lhsVal, mlir::Value rhsVal,
                                       RelationalOpKind opKind);

  mlir::Value getMLIRLogicalOp(mlir::Value lhsVal, mlir::Value rhsVal,
                               LogicalOpKind opKind);

  mlir::Block *getNewBlock(mlir::Block *insertBefore);

  FC::AllocaOp createAlloca(mlir::Type type, llvm::StringRef name);

  FC::AllocaOp createAlloca(Symbol *symbol);

  bool emitFunctionDeclaration(ProgramUnit *PU);

  bool runOnProgram(ParseTree *parseTree) override;

  bool updateArgForNestProgramUnit();

  FC::FCLoadOp emitLoadInstruction(mlir::Value V,
                                   const mlir::Twine &Name = "load",
                                   bool disableTBAA = false);

  void emitStoreInstruction(mlir::Value V, mlir::Value Ptr,
                            bool disableTBAA = false);

  mlir::Value getIntentArgForSymbol(Symbol *symbol);

  mlir::Value getArgumentFor(mlir::Value currArg, fc::Type *currArgTy,
                             fc::Type *fcDummyArgTy, FC::FCFuncOp *llFn,
                             unsigned argNum);

  mlir::Value getDynamicArrayFor(mlir::Value val, fc::ArrayType *staticArrTy,
                                 fc::ArrayType *dynArrTy);

  mlir::Location getLoc(fc::SourceLoc loc);

  mlir::SymbolRefAttr getIntrinFunction(llvm::StringRef name, mlir::Type type);

  mlir::SymbolRefAttr getFmaxFunction();

  bool createSubPUHelpers(ProgramUnit *PU);

  // OpenMP
  bool emitOpenMPParallelStmt(OpenMPParallelStmt *stmt);

  bool emitOpenMPParallelDoStmt(OpenMPParallelDoStmt *stmt);

  bool emitOpenMPSingleStmt(OpenMPSingleStmt *stmt);

  bool emitOpenMPMasterStmt(OpenMPMasterStmt *stmt);

  bool emitOpenMPDoStmt(OpenMPDoStmt *stmt);

  mlir::SymbolRefAttr getSymbolScopeList(Symbol *sym);

  mlir::Operation *getOpForSymRef(mlir::SymbolRefAttr symRef);

private:
  ParseTree *parseTree;
  ASTContext &C;
  mlir::MLIRContext &mlirContext;
  mlir::OpBuilder builder;
  mlir::OwningModuleRef &theModule;
  CGContext context;
  RuntimeHelper *runtimeHelper;
  CGASTHelper *cgHelper;
  Standard std;
  std::map<std::string, mlir::FunctionType> calledFuncs;
  bool EnableDebug;
};

} // namespace fc

#endif // FC_CODEGEN_H_
