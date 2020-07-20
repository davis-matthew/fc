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

#include "dialect/OpenMP/OpenMPOps.h"

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
static constexpr const char *kmpcSingleFnName = "__kmpc_single";
static constexpr const char *kmpcEndSingleFnName = "__kmpc_end_single";
static constexpr const char *kmpcThreadNumFnName = "__kmpc_global_thread_num";
static constexpr const char *kmpcBarrierFnName = "__kmpc_barrier";
static constexpr const char *kmpcMasterFnName = "__kmpc_master";
static constexpr const char *kmpcEndMasterFnName = "__kmpc_end_master";
// TODO: Suffix to be decided correctly
static constexpr const char *kmpcForStaticFnName = "__kmpc_for_static_init_4";
static constexpr const char *kmpcForStaticFiniFnName = "__kmpc_for_static_fini";

class OpenMPBuilder {
  mlir::ModuleOp &module;
  LLVM::LLVMDialect *llvmDialect;
  MLIRContext *context;

  LLVMType getIdentStructTy();

public:
  OpenMPBuilder(mlir::ModuleOp &mod, LLVM::LLVMDialect *_llvmDialect,
                MLIRContext *context)
      : module(mod), llvmDialect(_llvmDialect) {}

  LLVM::LLVMFuncOp getFunction(LLVMType fnTy, llvm::StringRef name,
                               mlir::OpBuilder &rewriter);

  LLVM::LLVMFuncOp getKmpcForkCall(mlir::OpBuilder &rewriter);

  LLVM::LLVMFuncOp getKmpcSingleFn(mlir::OpBuilder &rewriter);

  LLVM::LLVMFuncOp getKmpcEndSingleFn(mlir::OpBuilder &rewriter);

  LLVM::LLVMFuncOp getKmpcMasterFn(mlir::OpBuilder &rewriter);

  LLVM::LLVMFuncOp getKmpcEndMasterFn(mlir::OpBuilder &rewriter);

  LLVM::LLVMFuncOp getKmpcThreadNumFn(mlir::OpBuilder &rewriter);

  LLVM::LLVMFuncOp getKmpcBarrierFn(mlir::OpBuilder &rewriter);

  LLVM::LLVMFuncOp getKmpcForStatic(mlir::OpBuilder &rewriter);

  LLVM::LLVMFuncOp getKmpcForStaticFini(mlir::OpBuilder &rewriter);

  mlir::Value getGlobalIdent(mlir::Location loc, mlir::OpBuilder &rewriter);

  LLVMFuncOp createOutlinedFunction(mlir::Region *region,
                                    mlir::OpBuilder &rewriter,
                                    mlir::Location loc,
                                    llvm::ArrayRef<mlir::Value> args);

  LLVMFuncOp createOutlinedFunctionFor(OMP::ParallelDoOp op);

  void cloneConstantsFromParent(Region *region);

  LLVMType getVoidIdentIntFnTy();

  LLVMType getIntIdentIntFnTy();
};

/// \brief Create struct type for ident.
/// type { i32, i32, i32, i32, i8* }
LLVMType OpenMPBuilder::getIdentStructTy() {
  assert(context);
  auto I32 = LLVMType::getInt32Ty(llvmDialect);
  auto I8Ptr = LLVMType::getInt8PtrTy(llvmDialect);
  return LLVMType::getStructTy(llvmDialect, {I32, I32, I32, I32, I8Ptr});
}

/// \brief Create a function type of type (void (ident*, i32))
LLVMType OpenMPBuilder::getVoidIdentIntFnTy() {
  auto I32 = LLVMType::getInt32Ty(llvmDialect);
  auto identTy = getIdentStructTy().getPointerTo();
  auto voidTy = LLVMType::getVoidTy(llvmDialect);
  auto llvmFnType = LLVMType::getFunctionTy(voidTy, {identTy, I32},
                                            /*isVarArg=*/false);
  return llvmFnType;
}

/// \brief Create a function type of type (int (ident*, i32))
LLVMType OpenMPBuilder::getIntIdentIntFnTy() {
  auto I32 = LLVMType::getInt32Ty(llvmDialect);
  auto identTy = getIdentStructTy().getPointerTo();
  auto llvmFnType = LLVMType::getFunctionTy(I32, {identTy, I32},
                                            /*isVarArg=*/false);
  return llvmFnType;
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

  // FIXME : Currently these values are taken from clang generated ir
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

/// \brief Returns __kmpc_for_static_init_4 function
LLVM::LLVMFuncOp OpenMPBuilder::getKmpcForStatic(mlir::OpBuilder &rewriter) {
  if (auto funcOp = module.lookupSymbol<LLVMFuncOp>(kmpcForStaticFnName))
    return funcOp;

  auto I32 = LLVMType::getInt32Ty(llvmDialect);
  auto I32Ptr = I32.getPointerTo();
  auto identTy = getIdentStructTy().getPointerTo();
  auto voidTy = LLVMType::getVoidTy(llvmDialect);
  auto llvmFnType = LLVMType::getFunctionTy(
      voidTy, {identTy, I32, I32, I32Ptr, I32Ptr, I32Ptr, I32Ptr, I32, I32},
      /*isVarArg=*/true);
  return getFunction(llvmFnType, kmpcForStaticFnName, rewriter);
}

/// \brief Inserts and returns a function of name \p fnName and of typ
/// \p fnType
LLVM::LLVMFuncOp OpenMPBuilder::getFunction(LLVMType fnType,
                                            llvm::StringRef fnName,
                                            mlir::OpBuilder &rewriter) {
  mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVMFuncOp>(module.getLoc(), fnName, fnType);
  return module.lookupSymbol<LLVM::LLVMFuncOp>(fnName);
}

/// \brief Createas __kmpc_for_static_fini function
LLVM::LLVMFuncOp
OpenMPBuilder::getKmpcForStaticFini(mlir::OpBuilder &rewriter) {
  if (auto funcOp = module.lookupSymbol<LLVMFuncOp>(kmpcForStaticFiniFnName))
    return funcOp;

  auto llvmFnType = getVoidIdentIntFnTy();
  return getFunction(llvmFnType, kmpcForStaticFiniFnName, rewriter);
}

/// \brief Creates kmpc_fork_call function,
LLVM::LLVMFuncOp OpenMPBuilder::getKmpcForkCall(mlir::OpBuilder &rewriter) {

  auto fnName = kmpcForkFnName;
  if (auto funcOp = module.lookupSymbol<LLVMFuncOp>(fnName))
    return funcOp;

  auto I32 = LLVMType::getInt32Ty(llvmDialect);
  auto I8Ptr = LLVMType::getInt8PtrTy(llvmDialect);
  auto identTy = getIdentStructTy().getPointerTo();
  auto voidTy = LLVMType::getVoidTy(llvmDialect);
  auto llvmFnType = LLVMType::getFunctionTy(voidTy, {identTy, I32, I8Ptr},
                                            /*isVarArg=*/true);

  return getFunction(llvmFnType, fnName, rewriter);
}

/// \brief Creates kmpc_single function
LLVM::LLVMFuncOp OpenMPBuilder::getKmpcSingleFn(mlir::OpBuilder &rewriter) {

  if (auto funcOp = module.lookupSymbol<LLVMFuncOp>(kmpcSingleFnName))
    return funcOp;

  auto llvmFnType = getIntIdentIntFnTy();

  return getFunction(llvmFnType, kmpcSingleFnName, rewriter);
}

/// \brief Creates kmpc end single function
LLVM::LLVMFuncOp OpenMPBuilder::getKmpcEndSingleFn(mlir::OpBuilder &rewriter) {

  if (auto funcOp = module.lookupSymbol<LLVMFuncOp>(kmpcEndSingleFnName))
    return funcOp;

  auto llvmFnType = getVoidIdentIntFnTy();
  return getFunction(llvmFnType, kmpcEndMasterFnName, rewriter);
}

/// \brief Creates kmpc_master function
LLVM::LLVMFuncOp OpenMPBuilder::getKmpcMasterFn(mlir::OpBuilder &rewriter) {

  if (auto funcOp = module.lookupSymbol<LLVMFuncOp>(kmpcMasterFnName))
    return funcOp;

  auto llvmFnType = getIntIdentIntFnTy();
  return getFunction(llvmFnType, kmpcMasterFnName, rewriter);
}

/// \brief Creates kmpc end master function
LLVM::LLVMFuncOp OpenMPBuilder::getKmpcEndMasterFn(mlir::OpBuilder &rewriter) {

  if (auto funcOp = module.lookupSymbol<LLVMFuncOp>(kmpcEndMasterFnName))
    return funcOp;

  auto llvmFnType = getVoidIdentIntFnTy();
  mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVMFuncOp>(module.getLoc(), kmpcEndMasterFnName, llvmFnType);
  return module.lookupSymbol<LLVM::LLVMFuncOp>(kmpcEndMasterFnName);
}

/// \brief Creates __kmpc_barrier functions declaration
LLVM::LLVMFuncOp OpenMPBuilder::getKmpcBarrierFn(mlir::OpBuilder &rewriter) {

  if (auto funcOp = module.lookupSymbol<LLVMFuncOp>(kmpcBarrierFnName))
    return funcOp;

  auto I32 = LLVMType::getInt32Ty(llvmDialect);
  auto identTy = getIdentStructTy().getPointerTo();
  auto voidTy = LLVMType::getVoidTy(llvmDialect);
  auto llvmFnType = LLVMType::getFunctionTy(voidTy, {identTy, I32},
                                            /*isVarArg=*/false);

  return getFunction(llvmFnType, kmpcBarrierFnName, rewriter);
}

LLVM::LLVMFuncOp OpenMPBuilder::getKmpcThreadNumFn(mlir::OpBuilder &rewriter) {

  if (auto funcOp = module.lookupSymbol<LLVMFuncOp>(kmpcThreadNumFnName))
    return funcOp;

  auto identTy = getIdentStructTy().getPointerTo();
  auto I32 = LLVMType::getInt32Ty(llvmDialect);
  auto llvmFnType = LLVMType::getFunctionTy(I32, {identTy},
                                            /*isVarArg=*/false);

  return getFunction(llvmFnType, kmpcThreadNumFnName, rewriter);
}

/// \brief Creates outlined function of type
//          (void (i32*, i32*, arg1, arg2, ...))
LLVMFuncOp OpenMPBuilder::createOutlinedFunctionFor(OMP::ParallelDoOp op) {

  auto loc = op.getLoc();
  llvm::SmallVector<mlir::Value, 2> args(op.args());
  OpBuilder rewriter(op);

  // TODO Should be unique across application, what should be suffix ?
  //      Line number ?
  static int suffix = 1;
  std::string fnName = outlinedPrefix;
  fnName += std::to_string(suffix);
  suffix++;

  auto I32Ptr = LLVMType::getInt32Ty(llvmDialect).getPointerTo();
  llvm::SmallVector<LLVM::LLVMType, 2> types{I32Ptr, I32Ptr};
  for (auto arg : args) {
    if (arg.getType().cast<LLVMType>().isStructTy()) {
      types.push_back(arg.getType().cast<LLVMType>().getPointerTo());
      continue;
    }
    types.push_back(arg.getType().cast<LLVM::LLVMType>());
  }

  auto voidTy = LLVMType::getVoidTy(llvmDialect);
  auto llvmFnType = LLVMType::getFunctionTy(voidTy, types, false);

  rewriter.setInsertionPointToStart(module.getBody());
  auto outlinedFn =
      rewriter.create<LLVMFuncOp>(module.getLoc(), fnName, llvmFnType);

  // TODO : This is repeated code, reason being for block argument types
  //       should mlir::Type and for LLVM::Func argument types need to
  //       be LLVMType
  llvm::SmallVector<mlir::Type, 2> argTypes{I32Ptr, I32Ptr};
  for (auto arg : args) {
    if (arg.getType().cast<LLVMType>().isStructTy()) {
      argTypes.push_back(arg.getType().cast<LLVMType>().getPointerTo());
      continue;
    }
    argTypes.push_back(arg.getType());
  }

  auto newRegion = &outlinedFn.body();
  if (newRegion->empty()) {
    rewriter.createBlock(newRegion, newRegion->end());
  }

  auto entryBB = &outlinedFn.body().front();
  entryBB->addArguments(argTypes);

  unsigned k = 2;
  // First two arguments are builtin
  // First 4 arguments are namely
  // 0 -> IV
  // 1 -> LB of original loop
  // 2 -> UB of original loop
  // 3 -> Step of original loop
  auto oldAlloca = entryBB->getArgument(k++);
  mlir::Value lb = entryBB->getArgument(k++);
  mlir::Value ub = entryBB->getArgument(k++);
  mlir::Value step = entryBB->getArgument(k++);
  auto I32 = LLVMType::getInt32Ty(llvmDialect);
  lb = rewriter.create<LLVM::LoadOp>(loc, I32, lb);
  ub = rewriter.create<LLVM::LoadOp>(loc, I32, ub);
  step = rewriter.create<LLVM::LoadOp>(loc, I32, step);
  auto one = rewriter.create<LLVM::ConstantOp>(loc, I32,
                                               rewriter.getI32IntegerAttr(1));
  auto zero = rewriter.create<LLVM::ConstantOp>(loc, I32,
                                                rewriter.getI32IntegerAttr(0));

  // TODO : Need not be 32 always
  auto schedType = rewriter.create<LLVM::ConstantOp>(
      loc, I32, rewriter.getI32IntegerAttr(34));

  auto lbAlloca = rewriter.create<AllocaOp>(loc, I32Ptr, one, 1);
  auto ubAlloca = rewriter.create<AllocaOp>(loc, I32Ptr, one, 1);
  auto stepAlloca = rewriter.create<AllocaOp>(loc, I32Ptr, one, 1);
  auto lastierAlloca = rewriter.create<AllocaOp>(loc, I32Ptr, one, 1);
  auto threadUbAlloca = rewriter.create<AllocaOp>(loc, I32Ptr, one, 1);
  auto threadIVAlloca = rewriter.create<AllocaOp>(loc, I32Ptr, one, 1);

  rewriter.create<LLVM::StoreOp>(loc, lb, lbAlloca);
  rewriter.create<LLVM::StoreOp>(loc, ub, ubAlloca);
  rewriter.create<LLVM::StoreOp>(loc, step, stepAlloca);
  rewriter.create<LLVM::StoreOp>(loc, zero, lastierAlloca);

  // create required basic block
  auto newEntryBB = rewriter.createBlock(newRegion, newRegion->end());
  auto ubBB1 = rewriter.createBlock(newRegion, newRegion->end());
  auto ubBB2 = rewriter.createBlock(newRegion, newRegion->end());
  auto PH = rewriter.createBlock(newRegion, newRegion->end());
  auto header = rewriter.createBlock(newRegion, newRegion->end());
  auto latch = rewriter.createBlock(newRegion, newRegion->end());
  auto exitBB = rewriter.createBlock(newRegion, newRegion->end());
  auto retBB = rewriter.createBlock(newRegion, newRegion->end());

  rewriter.setInsertionPointToEnd(entryBB);

  auto entryCheck = rewriter.create<ICmpOp>(loc, ICmpPredicate::slt, zero, ub);

  rewriter.create<CondBrOp>(loc, ArrayRef<Value>{entryCheck},
                            ArrayRef<Block *>{newEntryBB, retBB});

  rewriter.setInsertionPointToEnd(newEntryBB);

  auto ident = getGlobalIdent(loc, rewriter);
  mlir::Value threadNum = entryBB->getArgument(0);
  threadNum = rewriter.create<LLVM::LoadOp>(loc, I32, threadNum);

  auto forStaticFn = getKmpcForStatic(rewriter);

  // Create call to __kmpc_for_static_init_4
  // arg0 -> ident
  // arg1 -> thread num
  // arg2 -> sched type, pass 34 currently
  // arg3 -> plastier
  // arg4 -> lower
  // arg5 -> upper
  // arg6 -> stride
  // arg7 -> incr
  // arg8 -> chunk
  rewriter.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{}, rewriter.getSymbolRefAttr(forStaticFn),
      ArrayRef<Value>{ident, threadNum, schedType, lastierAlloca, lbAlloca,
                      ubAlloca, stepAlloca, one, one});
  auto newUb = rewriter.create<LLVM::LoadOp>(loc, I32, ubAlloca);

  auto condOp = rewriter.create<ICmpOp>(loc, ICmpPredicate::sgt, newUb, ub);

  rewriter.create<CondBrOp>(loc, ArrayRef<Value>{condOp},
                            ArrayRef<Block *>{ubBB1, ubBB2});

  rewriter.setInsertionPointToEnd(ubBB1);
  rewriter.create<LLVM::StoreOp>(loc, ub, threadUbAlloca);

  rewriter.create<BrOp>(loc, ArrayRef<Value>{}, ArrayRef<Block *>{PH},
                        ValueRange());

  rewriter.setInsertionPointToEnd(ubBB2);
  newUb = rewriter.create<LLVM::LoadOp>(loc, I32, ubAlloca);
  rewriter.create<LLVM::StoreOp>(loc, newUb, threadUbAlloca);

  rewriter.create<BrOp>(loc, ArrayRef<Value>{}, ArrayRef<Block *>{PH},
                        ValueRange());

  rewriter.setInsertionPointToEnd(PH);

  // Initialize induction variable
  auto newLb = rewriter.create<LLVM::LoadOp>(loc, I32, lbAlloca);
  rewriter.create<LLVM::StoreOp>(loc, newLb, threadIVAlloca);
  rewriter.create<BrOp>(loc, ArrayRef<Value>{}, ArrayRef<Block *>{header},
                        ValueRange());

  rewriter.setInsertionPointToEnd(header);
  auto iv = rewriter.create<LLVM::LoadOp>(loc, I32, threadIVAlloca);
  newUb = rewriter.create<LLVM::LoadOp>(loc, I32, threadUbAlloca);

  auto region = &op.region();

  // Clone constants from parent if any
  cloneConstantsFromParent(region);

  auto front = &region->front();
  auto back = &region->back();
  newRegion->getBlocks().splice(++header->getIterator(), region->getBlocks(),
                                region->begin(), region->end());
  unsigned numArgs = front->getNumArguments();
  assert(numArgs == 0 && "Handle block arguments ");

  // iv <= ub br to body
  condOp = rewriter.create<ICmpOp>(loc, ICmpPredicate::sle, iv, newUb);
  rewriter.create<CondBrOp>(loc, ArrayRef<Value>{condOp},
                            ArrayRef<Block *>{front, exitBB});

  auto terminator = back->getTerminator();
  rewriter.setInsertionPoint(terminator);
  rewriter.create<BrOp>(loc, ArrayRef<Value>{}, ArrayRef<Block *>{latch},
                        ValueRange());
  terminator->erase();

  // create loop latch with backedge to header
  rewriter.setInsertionPointToStart(latch);
  iv = rewriter.create<LLVM::LoadOp>(loc, I32, threadIVAlloca);
  auto ivUpdate = rewriter.create<LLVM::AddOp>(loc, I32, iv, one);
  rewriter.create<LLVM::StoreOp>(loc, ivUpdate, threadIVAlloca);
  rewriter.create<BrOp>(loc, ArrayRef<Value>{}, ArrayRef<Block *>{header},
                        ValueRange());

  rewriter.setInsertionPointToStart(exitBB);

  ident = getGlobalIdent(loc, rewriter);
  threadNum = entryBB->getArgument(0);
  threadNum = rewriter.create<LLVM::LoadOp>(loc, I32, threadNum);
  auto forFiniFn = getKmpcForStaticFini(rewriter);
  rewriter.create<LLVM::CallOp>(loc, ArrayRef<Type>{},
                                rewriter.getSymbolRefAttr(forFiniFn),
                                ArrayRef<Value>{ident, threadNum});
  rewriter.create<BrOp>(loc, ArrayRef<Value>{}, ArrayRef<Block *>{retBB},
                        ValueRange());
  rewriter.setInsertionPointToStart(retBB);

  rewriter.create<LLVM::ReturnOp>(loc, llvm::ArrayRef<Value>{});

  // First two arguments are builtin
  k = 2;
  rewriter.setInsertionPointToStart(entryBB);
  for (auto arg : args) {
    if (arg.getType().cast<LLVMType>().isStructTy()) {
      auto newVal = rewriter.create<LLVM::LoadOp>(
          loc, arg.getType().cast<LLVMType>(), entryBB->getArgument(k++));
      replaceAllUsesInRegionWith(arg, newVal, *newRegion);
      continue;
    }
    replaceAllUsesInRegionWith(arg, entryBB->getArgument(k++), *newRegion);
  }

  // Finally replace uses of old IV
  replaceAllUsesInRegionWith(oldAlloca, threadIVAlloca, *newRegion);
  return outlinedFn;
}

/// \brief Creates outlined function of type
//          (void (i32*, i32*, arg1, arg2, ...))
LLVMFuncOp
OpenMPBuilder::createOutlinedFunction(Region *region, mlir::OpBuilder &rewriter,
                                      mlir::Location loc,
                                      llvm::ArrayRef<mlir::Value> args) {

  // TODO Should be unique, what should be suffix ?
  //      Line number ?
  static int suffix = 1;
  std::string fnName = outlinedPrefix;
  fnName += std::to_string(suffix);
  suffix++;
  auto I32Ptr = LLVMType::getInt32Ty(llvmDialect).getPointerTo();
  llvm::SmallVector<LLVM::LLVMType, 2> types{I32Ptr, I32Ptr};
  for (auto arg : args) {
    if (arg.getType().cast<LLVMType>().isStructTy()) {
      types.push_back(arg.getType().cast<LLVMType>().getPointerTo());
      continue;
    }
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
    if (arg.getType().cast<LLVMType>().isStructTy()) {
      argTypes.push_back(arg.getType().cast<LLVMType>().getPointerTo());
      continue;
    }
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
  auto newRegion = &outlinedFn.body();
  auto entryBB = &newRegion->front();
  rewriter.setInsertionPointToStart(entryBB);
  for (auto arg : args) {
    if (arg.getType().cast<LLVMType>().isStructTy()) {
      auto newVal = rewriter.create<LLVM::LoadOp>(
          loc, arg.getType().cast<LLVMType>(), entryBB->getArgument(k++));
      replaceAllUsesInRegionWith(arg, newVal, *newRegion);
      continue;
    }
    replaceAllUsesInRegionWith(arg, outlinedFn.body().front().getArgument(k++),
                               outlinedFn.body());
  }

  rewriter.restoreInsertionPoint(insertPoint);
  return outlinedFn;
}

void OpenMPBuilder::cloneConstantsFromParent(Region *region) {
  llvm::SmallVector<LLVM::ConstantOp, 2> constsToClone;
  for (auto &block : region->getBlocks()) {
    block.walk([&](mlir::Operation *op) {
      for (auto operand : op->getOperands()) {
        if (auto defOp = operand.getDefiningOp())
          if (auto Const = llvm::dyn_cast<LLVM::ConstantOp>(defOp))
            if (Const.getOperation()->getBlock()->getParent() != region)
              constsToClone.push_back(Const);
      }
    });
  }

  OpBuilder rewriter(region->getContext());
  rewriter.setInsertionPointToStart(&region->front());
  for (auto constant : constsToClone) {
    auto newConst = rewriter.clone(*constant.getOperation());
    replaceAllUsesInRegionWith(constant, newConst->getResult(0), *region);
  }
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
    module.walk([this](mlir::Operation *operation) {
      if (auto op = llvm::dyn_cast<OMP::ParallelOp>(operation)) {
        translateOmpParallelOp(op);
      }
      if (auto op = llvm::dyn_cast<OMP::SingleOp>(operation)) {
        translateOmpSingleOp(op);
      }
      if (auto op = llvm::dyn_cast<OMP::MasterOp>(operation)) {
        translateOmpMasterOp(op);
      }
      if (auto op = llvm::dyn_cast<OMP::OmpDoOp>(operation)) {
        translateOmpDoOp(op);
      }
      if (auto op = llvm::dyn_cast<OMP::ParallelDoOp>(operation)) {
        translateOmpParallelDoOp(op);
      }
    });
  }

  void translateOmpParallelOp(OMP::ParallelOp op);

  void translateOmpSingleOp(OMP::SingleOp op);

  void translateOmpMasterOp(OMP::MasterOp op);

  void translateOmpDoOp(OMP::OmpDoOp op);

  void translateOmpParallelDoOp(OMP::ParallelDoOp op);
};

// FIXME : Ideally we should use splitBlock() function here. If we use
//         Block::splitBlock(), iterator of original block points to
//         invalid iterator after insertpt is reached.
static Block *splitBlockAt(Block *block, Operation *op) {
  OpBuilder rewriter(op);
  auto newParentBB = rewriter.createBlock(block);
  for (auto arg : block->getArguments()) {
    newParentBB->addArgument(arg.getType());
  }

  auto k = 0;
  auto region = block->getParent();
  while (block->getNumArguments()) {
    replaceAllUsesInRegionWith(block->getArgument(0),
                               newParentBB->getArgument(k), *region);
    k++;
    block->eraseArgument(0);
  }

  newParentBB->getOperations().splice(newParentBB->end(),
                                      block->getOperations(), block->begin(),
                                      op->getIterator());
  return newParentBB;
}

void OpenMPLowering::translateOmpParallelDoOp(OMP::ParallelDoOp op) {
  auto loc = op.getLoc();

  OpBuilder rewriter(op);
  OpenMPBuilder builder(module, llvmDialect, context);
  llvm::SmallVector<mlir::Value, 2> args(op.args());

  auto outlinedFunc = builder.createOutlinedFunctionFor(op);

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

  auto one = rewriter.create<LLVM::ConstantOp>(loc, I32,
                                               rewriter.getI32IntegerAttr(1));
  llvm::SmallVector<mlir::Value, 2> funcArgs{ident, count, finalPtr};
  for (auto arg : args) {
    if (arg.getType().cast<LLVMType>().isStructTy()) {
      auto alloca = rewriter.create<AllocaOp>(
          loc, arg.getType().cast<LLVMType>().getPointerTo(), one, 1);
      rewriter.create<LLVM::StoreOp>(loc, arg, alloca);
      funcArgs.push_back(alloca);
      continue;
    }
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

// Get the bounds for current thread from __kmpc_for_static_init and
// emit loop
void OpenMPLowering::translateOmpDoOp(OMP::OmpDoOp op) {

  auto loc = op.getLoc();
  // auto theFunction = op.getParentOfType<LLVMFuncOp>();
  OpBuilder rewriter(op);
  OpenMPBuilder builder(module, llvmDialect, context);

  auto lb = op.getLowerBound();
  auto ub = op.getUpperBound();
  auto step = op.getStep();

  auto I32 = LLVMType::getInt32Ty(llvmDialect);
  auto one = rewriter.create<LLVM::ConstantOp>(loc, I32,
                                               rewriter.getI32IntegerAttr(1));
  auto zero = rewriter.create<LLVM::ConstantOp>(loc, I32,
                                                rewriter.getI32IntegerAttr(0));

  auto schedType = rewriter.create<LLVM::ConstantOp>(
      loc, I32, rewriter.getI32IntegerAttr(34));
  auto I32Ptr = LLVMType::getInt32Ty(llvmDialect).getPointerTo();

  // rewriter.setInsertionPointToStart(&theFunction.body().front());
  auto lbAlloca = rewriter.create<AllocaOp>(loc, I32Ptr, one, 1);
  auto ubAlloca = rewriter.create<AllocaOp>(loc, I32Ptr, one, 1);
  auto stepAlloca = rewriter.create<AllocaOp>(loc, I32Ptr, one, 1);
  auto lastierAlloca = rewriter.create<AllocaOp>(loc, I32Ptr, one, 1);
  auto threadUbAlloca = rewriter.create<AllocaOp>(loc, I32Ptr, one, 1);
  auto threadIVAlloca = rewriter.create<AllocaOp>(loc, I32Ptr, one, 1);

  rewriter.create<LLVM::StoreOp>(loc, lb, lbAlloca);
  rewriter.create<LLVM::StoreOp>(loc, ub, ubAlloca);
  rewriter.create<LLVM::StoreOp>(loc, step, stepAlloca);
  rewriter.create<LLVM::StoreOp>(loc, zero, lastierAlloca);

  auto ident = builder.getGlobalIdent(loc, rewriter);
  auto threadNumFn = builder.getKmpcThreadNumFn(rewriter);
  auto threadNumOp = rewriter.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{I32}, rewriter.getSymbolRefAttr(threadNumFn),
      ArrayRef<Value>{ident});

  auto forStaticFn = builder.getKmpcForStatic(rewriter);

  // Create call to __kmpc_for_static_init_4
  // arg0 -> ident
  // arg1 -> thread num
  // arg2 -> sched type, pass 34 currently
  // arg3 -> plastier
  // arg4 -> lower
  // arg5 -> upper
  // arg6 -> stride
  // arg7 -> incr
  // arg8 -> chunk
  rewriter.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{}, rewriter.getSymbolRefAttr(forStaticFn),
      ArrayRef<Value>{ident, threadNumOp.getResult(0), schedType, lastierAlloca,
                      lbAlloca, ubAlloca, stepAlloca, one, one});

  auto block = op.getOperation()->getBlock();
  // auto parentRegion = block->getParent();
  auto newParentBB = splitBlockAt(block, op.getOperation());
  rewriter.setInsertionPointToEnd(newParentBB);

  auto newUb = rewriter.create<LLVM::LoadOp>(loc, I32, ubAlloca);

  auto condOp = rewriter.create<ICmpOp>(loc, ICmpPredicate::sgt, newUb, ub);

  // create required basic block
  auto ubBB1 = rewriter.createBlock(block);
  auto ubBB2 = rewriter.createBlock(block);
  auto PH = rewriter.createBlock(block);
  auto header = rewriter.createBlock(block);
  auto latch = rewriter.createBlock(block);

  // compute the upperbound for current thread and store it in threadUbAlloca
  rewriter.setInsertionPointToEnd(newParentBB);
  rewriter.create<CondBrOp>(loc, ArrayRef<Value>{condOp},
                            ArrayRef<Block *>{ubBB1, ubBB2});

  rewriter.setInsertionPointToEnd(ubBB1);
  rewriter.create<LLVM::StoreOp>(loc, ub, threadUbAlloca);

  rewriter.create<BrOp>(loc, ArrayRef<Value>{}, ArrayRef<Block *>{PH},
                        ValueRange());

  rewriter.setInsertionPointToEnd(ubBB2);
  newUb = rewriter.create<LLVM::LoadOp>(loc, I32, ubAlloca);
  rewriter.create<LLVM::StoreOp>(loc, newUb, threadUbAlloca);

  rewriter.create<BrOp>(loc, ArrayRef<Value>{}, ArrayRef<Block *>{PH},
                        ValueRange());

  rewriter.setInsertionPointToEnd(PH);

  // Initialize induction variable
  auto newLb = rewriter.create<LLVM::LoadOp>(loc, I32, lbAlloca);
  rewriter.create<LLVM::StoreOp>(loc, newLb, threadIVAlloca);
  rewriter.create<BrOp>(loc, ArrayRef<Value>{}, ArrayRef<Block *>{header},
                        ValueRange());

  rewriter.setInsertionPointToEnd(header);
  auto iv = rewriter.create<LLVM::LoadOp>(loc, I32, threadIVAlloca);
  newUb = rewriter.create<LLVM::LoadOp>(loc, I32, threadUbAlloca);

  auto region = &op.region();
  replaceAllUsesInRegionWith(op.getIndVar(), threadIVAlloca, *region);
  auto front = &region->front();
  auto back = &region->back();

  block->getParent()->getBlocks().splice(++header->getIterator(),
                                         region->getBlocks(), region->begin(),
                                         region->end());

  // TODO : Add iv replacement stuff
  unsigned numArgs = front->getNumArguments();
  for (unsigned i = 0; i < numArgs; ++i) {
    front->eraseArgument(i);
  }

  // iv <= ub br to body
  condOp = rewriter.create<ICmpOp>(loc, ICmpPredicate::sle, iv, newUb);
  rewriter.create<CondBrOp>(loc, ArrayRef<Value>{condOp},
                            ArrayRef<Block *>{front, block});

  auto terminator = back->getTerminator();
  rewriter.setInsertionPoint(terminator);
  rewriter.create<BrOp>(loc, ArrayRef<Value>{}, ArrayRef<Block *>{latch},
                        ValueRange());
  terminator->erase();

  // create loop latch with backedge to header
  rewriter.setInsertionPointToStart(latch);
  iv = rewriter.create<LLVM::LoadOp>(loc, I32, threadIVAlloca);
  auto ivUpdate = rewriter.create<LLVM::AddOp>(loc, I32, iv, one);
  rewriter.create<LLVM::StoreOp>(loc, ivUpdate, threadIVAlloca);
  rewriter.create<BrOp>(loc, ArrayRef<Value>{}, ArrayRef<Block *>{header},
                        ValueRange());

  rewriter.setInsertionPointToStart(block);
  auto forFiniFn = builder.getKmpcForStaticFini(rewriter);
  rewriter.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{}, rewriter.getSymbolRefAttr(forFiniFn),
      ArrayRef<Value>{ident, threadNumOp.getResult(0)});

  auto barrierFn = builder.getKmpcBarrierFn(rewriter);
  rewriter.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{}, rewriter.getSymbolRefAttr(barrierFn),
      ArrayRef<Value>{ident, threadNumOp.getResult(0)});

  op.erase();
}

// step 1 : Call to __kmpc_single()
// step 2 : If it returns zero execute the body of \p op
// step 3 : At the end call __kmpc_end_single
// step 4 : In the exit block call __kmpc_barrier.
void OpenMPLowering::translateOmpSingleOp(OMP::SingleOp op) {
  // auto theFunction = op.getParentOfType<LLVMFuncOp>();
  // theFunction.dump();
  auto loc = op.getLoc();
  auto region = op.getRegion();
  llvm::SmallVector<mlir::Value, 2> args(op.symbols());

  OpBuilder rewriter(op);
  auto block = op.getOperation()->getBlock();

  auto newParentBB = splitBlockAt(block, op.getOperation());

  rewriter.setInsertionPointToEnd(newParentBB);
  auto I32 = LLVMType::getInt32Ty(llvmDialect);
  auto zero = rewriter.create<LLVM::ConstantOp>(loc, I32,
                                                rewriter.getI32IntegerAttr(0));
  OpenMPBuilder builder(module, llvmDialect, context);
  auto ident = builder.getGlobalIdent(loc, rewriter);
  auto threadNumFn = builder.getKmpcThreadNumFn(rewriter);
  auto threadNumOp = rewriter.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{I32}, rewriter.getSymbolRefAttr(threadNumFn),
      ArrayRef<Value>{ident});
  auto singleFn = builder.getKmpcSingleFn(rewriter);
  auto singleFnCall = rewriter.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{I32}, rewriter.getSymbolRefAttr(singleFn),
      ArrayRef<Value>{ident, threadNumOp.getResult(0)});

  auto condOp = rewriter.create<ICmpOp>(loc, ICmpPredicate::ne,
                                        singleFnCall.getResult(0), zero);

  auto front = &region->front();
  auto back = &region->back();

  block->getParent()->getBlocks().splice(++newParentBB->getIterator(),
                                         region->getBlocks(), region->begin(),
                                         region->end());

  rewriter.create<CondBrOp>(loc, ArrayRef<Value>{condOp},
                            ArrayRef<Block *>{front, block});

  auto terminator = back->getTerminator();
  rewriter.setInsertionPoint(terminator);
  auto endSingleFn = builder.getKmpcEndSingleFn(rewriter);
  rewriter.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{}, rewriter.getSymbolRefAttr(endSingleFn),
      ArrayRef<Value>{ident, threadNumOp.getResult(0)});
  rewriter.create<BrOp>(loc, ArrayRef<Value>{}, ArrayRef<Block *>{block},
                        ValueRange());
  terminator->erase();

  rewriter.setInsertionPointToStart(block);
  auto barrierFn = builder.getKmpcBarrierFn(rewriter);
  rewriter.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{}, rewriter.getSymbolRefAttr(barrierFn),
      ArrayRef<Value>{ident, threadNumOp.getResult(0)});
  op.erase();
}

// step 1 : Call to __kmpc_master()
// step 2 : If it returns zero execute the body of \p op
// step 3 : At the end call __kmpc_end_master
void OpenMPLowering::translateOmpMasterOp(OMP::MasterOp op) {
  // auto theFunction = op.getParentOfType<LLVMFuncOp>();
  // theFunction.dump();
  auto loc = op.getLoc();
  auto region = op.getRegion();
  llvm::SmallVector<mlir::Value, 2> args(op.symbols());

  OpBuilder rewriter(op);
  auto block = op.getOperation()->getBlock();
  auto newParentBB = splitBlockAt(block, op.getOperation());

  rewriter.setInsertionPointToEnd(newParentBB);
  auto I32 = LLVMType::getInt32Ty(llvmDialect);
  auto zero = rewriter.create<LLVM::ConstantOp>(loc, I32,
                                                rewriter.getI32IntegerAttr(0));
  OpenMPBuilder builder(module, llvmDialect, context);
  auto ident = builder.getGlobalIdent(loc, rewriter);
  auto threadNumFn = builder.getKmpcThreadNumFn(rewriter);
  auto threadNumOp = rewriter.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{I32}, rewriter.getSymbolRefAttr(threadNumFn),
      ArrayRef<Value>{ident});
  auto masterFn = builder.getKmpcMasterFn(rewriter);
  auto masterFnCall = rewriter.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{I32}, rewriter.getSymbolRefAttr(masterFn),
      ArrayRef<Value>{ident, threadNumOp.getResult(0)});

  auto condOp = rewriter.create<ICmpOp>(loc, ICmpPredicate::ne,
                                        masterFnCall.getResult(0), zero);

  auto front = &region->front();
  auto back = &region->back();

  block->getParent()->getBlocks().splice(++newParentBB->getIterator(),
                                         region->getBlocks(), region->begin(),
                                         region->end());

  rewriter.create<CondBrOp>(loc, ArrayRef<Value>{condOp},
                            ArrayRef<Block *>{front, block});

  auto terminator = back->getTerminator();
  rewriter.setInsertionPoint(terminator);
  auto endMasterFn = builder.getKmpcEndMasterFn(rewriter);
  rewriter.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{}, rewriter.getSymbolRefAttr(endMasterFn),
      ArrayRef<Value>{ident, threadNumOp.getResult(0)});
  rewriter.create<BrOp>(loc, ArrayRef<Value>{}, ArrayRef<Block *>{block},
                        ValueRange());
  terminator->erase();

  op.erase();
}

void OpenMPLowering::translateOmpParallelOp(OMP::ParallelOp op) {
  auto loc = op.getLoc();
  auto region = op.getRegion();
  llvm::SmallVector<mlir::Value, 2> args(op.symbols());

  OpBuilder rewriter(op);
  OpenMPBuilder builder(module, llvmDialect, context);

  // Clone constants in any
  builder.cloneConstantsFromParent(region);

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
