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
#include "dialect/OpenMPOps/OpenMPOps.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/Sequence.h"

#include "common/Debug.h"

using namespace mlir;
using namespace mlir::LLVM;

static constexpr const char *kmpcForkFnName = "__kmpc_fork_call";
static constexpr const char *identString = ";unknown;unknown;0;0;;\00";
static constexpr const char *outlinedPrefix = "outlined.";

class OpenMPBuilder {
  mlir::ModuleOp &module;
  LLVM::LLVMDialect *llvmDialect;
  MLIRContext *context;

  LLVMType getIdentStructTy();

public:
  OpenMPBuilder(mlir::ModuleOp &mod, LLVM::LLVMDialect *_llvmDialect,
                MLIRContext *context)
      : module(mod), llvmDialect(_llvmDialect) {}

  LLVM::LLVMFuncOp getKmpcForkCall(mlir::OpBuilder &rewriter);

  mlir::Value getGlobalIdent(mlir::Location loc, mlir::OpBuilder &rewriter);

  LLVMFuncOp createOutlinedFunction(mlir::Region *region,
                                    mlir::OpBuilder &rewriter,
                                    mlir::Location loc,
                                    llvm::ArrayRef<mlir::Value> args);
};

/// \brief Create struct type for ident.
/// type { i32, i32, i32, i32, i8* }
LLVMType OpenMPBuilder::getIdentStructTy() {
  assert(context);
  auto I32 = LLVMType::getInt32Ty(llvmDialect);
  auto I8Ptr = LLVMType::getInt8PtrTy(llvmDialect);
  return LLVMType::getStructTy(llvmDialect, {I32, I32, I32, I32, I8Ptr});
}

/// \brief Search for global variable of ident type. If found  return it,
/// if not found create and return.
mlir::Value OpenMPBuilder::getGlobalIdent(mlir::Location loc,
                                          mlir::OpBuilder &rewriter) {
  auto structTy = getIdentStructTy();
  auto identName = "ident.global";
  auto identGlobal = module.lookupSymbol<LLVM::GlobalOp>(identName);
  if (identGlobal)
    return rewriter.create<LLVM::AddressOfOp>(loc, identGlobal);

  mlir::OpBuilder::InsertPoint insertPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(module.getBody());
  identGlobal = rewriter.create<LLVM::GlobalOp>(loc, structTy,
                                                /*isConstant=*/true,
                                                LLVM::Linkage::Internal,
                                                identName, mlir::Attribute());

  mlir::Region *initRegion = &identGlobal.getInitializerRegion();
  assert(initRegion);
  mlir::Block *body = identGlobal.getInitializerBlock();

  if (!body) {
    body = rewriter.createBlock(initRegion);
  }
  assert(body);
  rewriter.setInsertionPointToStart(body);
  auto I32 = LLVMType::getInt32Ty(llvmDialect);
  auto zero = rewriter.create<LLVM::ConstantOp>(loc, I32,
                                                rewriter.getI32IntegerAttr(0));
  auto two = rewriter.create<LLVM::ConstantOp>(loc, I32,
                                               rewriter.getI32IntegerAttr(2));
  mlir::Value stringValue = createGlobalString(
      loc, rewriter, "ident.str", identString, Linkage::Internal, llvmDialect);

  Value init = rewriter.create<LLVM::UndefOp>(loc, structTy);

  // FIXME : Currently these values are taken from clang genrated ir
  init = rewriter.create<LLVM::InsertValueOp>(loc, structTy, init, zero,
                                              rewriter.getI64ArrayAttr(0));
  init = rewriter.create<LLVM::InsertValueOp>(loc, structTy, init, two,
                                              rewriter.getI64ArrayAttr(1));
  init = rewriter.create<LLVM::InsertValueOp>(loc, structTy, init, zero,
                                              rewriter.getI64ArrayAttr(2));
  init = rewriter.create<LLVM::InsertValueOp>(loc, structTy, init, zero,
                                              rewriter.getI64ArrayAttr(3));
  init = rewriter.create<LLVM::InsertValueOp>(loc, structTy, init, stringValue,
                                              rewriter.getI64ArrayAttr(4));

  rewriter.create<LLVM::ReturnOp>(loc, llvm::ArrayRef<mlir::Value>{init});

  rewriter.restoreInsertionPoint(insertPoint);
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, identGlobal);
  return globalPtr;
}

/// \brief Creates kmpc_fork_call function,
LLVM::LLVMFuncOp OpenMPBuilder::getKmpcForkCall(mlir::OpBuilder &rewriter) {

  auto fnName = kmpcForkFnName;
  if (module.lookupSymbol<LLVMFuncOp>(fnName))
    return module.lookupSymbol<LLVM::LLVMFuncOp>(fnName);

  auto I32 = LLVMType::getInt32Ty(llvmDialect);
  auto I8Ptr = LLVMType::getInt8PtrTy(llvmDialect);
  auto identTy = getIdentStructTy().getPointerTo();
  auto voidTy = LLVMType::getVoidTy(llvmDialect);
  auto llvmFnType = LLVMType::getFunctionTy(voidTy, {identTy, I32, I8Ptr},
                                            /*isVarArg=*/true);

  mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVMFuncOp>(module.getLoc(), fnName, llvmFnType);
  return module.lookupSymbol<LLVM::LLVMFuncOp>(fnName);
}

/// \brief Creates outlined function of type
//          (void (i32*, i32*, arg1, arg2, ...))
LLVMFuncOp
OpenMPBuilder::createOutlinedFunction(Region *region, mlir::OpBuilder &rewriter,
                                      mlir::Location loc,
                                      llvm::ArrayRef<mlir::Value> args) {

  // TODO Should be unique, what should be suffix ?
  //      Line number ?
  auto fnName = outlinedPrefix;
  auto I32Ptr = LLVMType::getInt32Ty(llvmDialect).getPointerTo();
  llvm::SmallVector<LLVM::LLVMType, 2> types{I32Ptr, I32Ptr};
  for (auto arg : args) {
    types.push_back(arg.getType().cast<LLVM::LLVMType>());
  }

  auto voidTy = LLVMType::getVoidTy(llvmDialect);
  auto llvmFnType = LLVMType::getFunctionTy(voidTy, types, false);

  mlir::OpBuilder::InsertPoint insertPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(module.getBody());
  auto outlinedFn =
      rewriter.create<LLVMFuncOp>(module.getLoc(), fnName, llvmFnType);

  llvm::SmallVector<mlir::Type, 2> argTypes{I32Ptr, I32Ptr};
  for (auto arg : args) {
    argTypes.push_back(arg.getType());
  }

  outlinedFn.body().takeBody(*region);
  outlinedFn.body().front().addArguments(argTypes);
  outlinedFn.body().walk([&](OMP::OpenMPTerminatorOp op) {
    rewriter.setInsertionPoint(op);
    rewriter.create<LLVM::ReturnOp>(loc, llvm::ArrayRef<Value>{});
    op.erase();
  });

  unsigned k = 2;
  for (auto arg : args) {
    replaceAllUsesInRegionWith(arg, outlinedFn.body().front().getArgument(k++),
                               outlinedFn.body());
  }

  rewriter.restoreInsertionPoint(insertPoint);
  return outlinedFn;
}

class OpenMPLowering : public ModulePass<OpenMPLowering> {

private:
  LLVM::LLVMDialect *llvmDialect;
  ModuleOp module;
  MLIRContext *context;

public:
  void runOnModule() override {
    llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    module = getModule();
    context = &getContext();
    getModule().walk([this](OMP::ParallelOp op) { translateOpenMPOp(op); });
  }

  void translateOpenMPOp(OMP::ParallelOp);
};

void OpenMPLowering::translateOpenMPOp(OMP::ParallelOp op) {
  auto loc = op.getLoc();
  auto region = op.getRegion();
  llvm::SmallVector<mlir::Value, 2> args(op.symbols());

  OpBuilder rewriter(op);
  OpenMPBuilder builder(module, llvmDialect, context);
  auto outlinedFunc =
      builder.createOutlinedFunction(region, rewriter, loc, args);
  auto kmpcFunc = builder.getKmpcForkCall(rewriter);
  auto ident = builder.getGlobalIdent(loc, rewriter);
  auto I32 = LLVMType::getInt32Ty(llvmDialect);
  auto I8Ptr = LLVMType::getInt8PtrTy(llvmDialect);
  rewriter.setInsertionPointAfter(op);
  auto count = rewriter.create<LLVM::ConstantOp>(
      loc, I32, rewriter.getI32IntegerAttr(args.size()));
  auto funcPtr = rewriter.create<LLVM::ConstantOp>(
      loc, outlinedFunc.getType(), rewriter.getSymbolRefAttr(outlinedFunc));
  auto finalPtr = rewriter.create<LLVM::BitcastOp>(
      loc, I8Ptr, ArrayRef<mlir::Value>(funcPtr));

  llvm::SmallVector<mlir::Value, 2> funcArgs{ident, count, finalPtr};
  for (auto arg : args) {
    funcArgs.push_back(arg);
  }

  // Creates call to kmpc_fork_call,with following arguments
  // arg0 -> struct.ident_t*
  // arg1 -> Number of symbols passed to omp region
  // arg2 -> Function pointer to outlined function
  // arg3, arg4 ... -> Arguments to outlined function
  rewriter.create<LLVM::CallOp>(loc, ArrayRef<Type>{},
                                rewriter.getSymbolRefAttr(kmpcFunc), funcArgs);

  op.erase();
}

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>> createOpenMPLoweringPass() {
  return std::make_unique<OpenMPLowering>();
}

static mlir::PassRegistration<OpenMPLowering> pass("lower-openmp",
                                                   "Openmp lowering pass");
