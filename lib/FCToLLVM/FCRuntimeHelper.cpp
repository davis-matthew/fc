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
#include "FCToLLVM/FCRuntimeHelper.h"
#include "common/Debug.h"

#include "llvm/IR/Module.h"

extern "C" {
#include "runtime/fc_runtime.h"
}

using namespace fc;

static IOTypeKind getReadTypeKind(mlir::Type *Ty) {

  auto ty = Ty->dyn_cast<mlir::LLVM::LLVMType>();
  assert(ty);
  assert(ty.isPointerTy());
  ty = ty.getPointerElementTy();
  if (ty.isIntegerTy(1)) {
    return IOTypeKind::int1;
  }

  if (ty.isIntegerTy(8)) {
    return IOTypeKind::int8;
  }
  if (ty.isIntegerTy(32)) {
    return IOTypeKind::int32;
  }
  if (ty.isIntegerTy(64)) {
    return IOTypeKind::int64;
  }
  if (ty.isIntegerTy(128)) {
    return IOTypeKind::int128;
  }
  if (ty.isFloatTy()) {
    return IOTypeKind::float32;
  }
  if (ty.isDoubleTy()) {
    return IOTypeKind::double_precision;
  }

  if (ty.isPointerTy() && ty.getPointerElementTy().isIntegerTy(8)) {
    return IOTypeKind::string;
  }

  ty.dump();
  llvm_unreachable("unhandled IOTypeKind");
}

static IOTypeKind getPrintTypeKind(mlir::Type Ty) {

  auto ty = Ty.dyn_cast<mlir::LLVM::LLVMType>();
  if (!ty) {
    if (Ty.isInteger(1)) {
      return IOTypeKind::int1;
    }
    if (Ty.isInteger(8)) {
      return IOTypeKind::int8;
    }
    if (Ty.isInteger(32)) {
      return IOTypeKind::int32;
    }
    if (Ty.isInteger(64)) {
      return IOTypeKind::int64;
    }
    if (Ty.isInteger(128)) {
      return IOTypeKind::int128;
    }
    if (Ty.isF32()) {
      return IOTypeKind::float32;
    }
    if (Ty.isF64()) {
      return IOTypeKind::double_precision;
    }
    if (Ty.isIndex()) {
      return IOTypeKind::int32;
    }
    if (auto cTy = Ty.dyn_cast<mlir::ComplexType>()) {
      if (cTy.getElementType().isF32())
        return IOTypeKind::complex_float;
      if (cTy.getElementType().isF64())
        return IOTypeKind::complex_double;
    }
  }

  assert(ty);
  if (ty.isIntegerTy(1)) {
    return IOTypeKind::int1;
  }

  if (ty.isIntegerTy(8)) {
    return IOTypeKind::int8;
  }
  if (ty.isIntegerTy(32)) {
    return IOTypeKind::int32;
  }
  if (ty.isIntegerTy(64)) {
    return IOTypeKind::int64;
  }
  if (ty.isIntegerTy(128)) {
    return IOTypeKind::int128;
  }
  if (ty.isFloatTy()) {
    return IOTypeKind::float32;
  }
  if (ty.isDoubleTy()) {
    return IOTypeKind::double_precision;
  }

  if (ty.isPointerTy() && ty.getPointerElementTy().isIntegerTy(8)) {
    return IOTypeKind::string;
  }

  // This is the complex type! Array types are passed with array info.
  if (ty.isArrayTy() &&
      (ty.getArrayElementType().isFloatTy() ||
       ty.getArrayElementType().isDoubleTy()) &&
      ty.getArrayNumElements() == 2) {
    auto eleTy = ty.getArrayElementType();
    return eleTy.isFloatTy() ? IOTypeKind::complex_float
                             : IOTypeKind::complex_double;
  }

  ty.dump();
  llvm_unreachable("unhandled IOTypeKind");
}

void RuntimeHelper::fillPrintArgsFor(
    mlir::Value val, llvm::SmallVectorImpl<mlir::Value> &argsList,
    mlir::Location loc, mlir::Value arrDimSize, mlir::Type baseType,
    mlir::ConversionPatternRewriter *IRB, bool isDynArr, bool isString) {

  auto I32 = IRB->getIntegerType(32);
  mlir::Type Ty = val.getType();

  IOTypeKind typeKind;
  if (isString)
    typeKind = string;
  else if (arrDimSize)
    typeKind = IOTypeKind::array;
  else
    typeKind = getPrintTypeKind(Ty);

  // 1. push the element type.
  argsList.push_back(
      IRB->create<mlir::ConstantIntOp>(loc, ((int)typeKind), I32));

  // If scalar, just send the val.
  if (typeKind != IOTypeKind::array) {
    // 2 Push the value in case of scalar and return.
    if (typeKind == IOTypeKind::float32) {
      val = IRB->create<mlir::LLVM::FPExtOp>(
          loc, mlir::LLVM::LLVMType::getDoubleTy(llvmDialect),
          llvm::ArrayRef<mlir::Value>{val});
    }

    if (typeKind == IOTypeKind::int1) {
      val = IRB->create<mlir::LLVM::SExtOp>(
          loc, mlir::LLVM::LLVMType::getInt32Ty(llvmDialect),
          llvm::ArrayRef<mlir::Value>{val});
    }
    argsList.push_back(val);
    return;
  }

  IOTypeKind baseKind = getPrintTypeKind(baseType);
  assert(baseKind != IOTypeKind::array);

  // 2. Push the base element type of array.
  argsList.push_back(
      IRB->create<mlir::ConstantIntOp>(loc, ((int)baseKind), I32));

  // 3. Push the dimensions
  argsList.push_back(arrDimSize);

  auto BC = IRB->create<mlir::LLVM::BitcastOp>(
      loc, mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect), val);
  argsList.push_back(BC);
}

void RuntimeHelper::fillReadArgsFor(
    mlir::Value val, llvm::SmallVectorImpl<mlir::Value> &argsList,
    mlir::Location loc, mlir::Value arrDimSize, mlir::Type baseType,
    mlir::ConversionPatternRewriter *IRB, bool isString) {

  auto I32 = IRB->getIntegerType(32);
  mlir::Type Ty = val.getType();

  IOTypeKind typeKind;
  if (isString)
    typeKind = IOTypeKind::string;
  else if (arrDimSize)
    typeKind = IOTypeKind::array;
  else
    typeKind = getReadTypeKind(&Ty); // TODO: have one getIOKind() api ?

  // 1. push the element type.
  argsList.push_back(
      IRB->create<mlir::ConstantIntOp>(loc, ((int)typeKind), I32));

  // If scalar, just send the val.
  if (typeKind != IOTypeKind::array) {
    // 2 Push the value in case of scalar and return.
    argsList.push_back(val);
    return;
  }

  // if array, do more.
  // Get the base kind

  IOTypeKind baseKind = getPrintTypeKind(baseType);
  assert(baseKind != IOTypeKind::array);

  // 2. Push the base element type of array.
  argsList.push_back(
      IRB->create<mlir::ConstantIntOp>(loc, ((int)baseKind), I32));

  // 3. Push the dimensions
  argsList.push_back(arrDimSize);

  // 4. Push the actual array base.

  auto BC = IRB->create<mlir::LLVM::BitcastOp>(
      loc, mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect), val);
  argsList.push_back(BC);
}

mlir::SymbolRefAttr
RuntimeHelper::getPrintFunction(mlir::PatternRewriter &rewriter) {

  if (M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(printFnName))
    return mlir::SymbolRefAttr::get(printFnName, context);

  auto llvmI32Ty = mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
  auto llvmVoidTy = mlir::LLVM::LLVMType::getVoidTy(llvmDialect);
  auto llvmFnType = mlir::LLVM::LLVMType::getFunctionTy(llvmVoidTy, {llvmI32Ty},
                                                        /*isVarArg=*/true);

  mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(M.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(M.getLoc(), printFnName, llvmFnType);
  return mlir::SymbolRefAttr::get(printFnName, context);
}

mlir::SymbolRefAttr
RuntimeHelper::getOpenFunction(mlir::PatternRewriter &rewriter) {
  if (M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(openFnName))
    return mlir::SymbolRefAttr::get(openFnName, context);

  auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  auto I32 = mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
  auto FnTy = mlir::LLVM::LLVMType::getFunctionTy(I32, {I32, I8Ptr},
                                                  /* isVarArg=*/false);

  mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(M.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(M.getLoc(), openFnName, FnTy);
  return mlir::SymbolRefAttr::get(openFnName, context);
}

mlir::SymbolRefAttr
RuntimeHelper::getCloseFunction(mlir::PatternRewriter &rewriter) {

  if (M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(fileCloseFnName))
    return mlir::SymbolRefAttr::get(fileCloseFnName, context);

  auto I32 = mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
  auto FnTy = mlir::LLVM::LLVMType::getFunctionTy(I32, {I32},
                                                  /* isVarArg=*/false);

  mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(M.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(M.getLoc(), fileCloseFnName, FnTy);
  return mlir::SymbolRefAttr::get(fileCloseFnName, context);
}

mlir::SymbolRefAttr
RuntimeHelper::getFileWriteFunction(mlir::PatternRewriter &rewriter) {
  if (M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(fileWriteFnName))
    return mlir::SymbolRefAttr::get(fileWriteFnName, context);

  auto llvmI32Ty = mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
  auto llvmVoidTy = mlir::LLVM::LLVMType::getVoidTy(llvmDialect);
  auto llvmFnType = mlir::LLVM::LLVMType::getFunctionTy(
      llvmVoidTy, {llvmI32Ty, llvmI32Ty}, /*isVarArg=*/true);

  mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(M.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(M.getLoc(), fileWriteFnName,
                                          llvmFnType);
  return mlir::SymbolRefAttr::get(fileWriteFnName, context);
}

mlir::SymbolRefAttr
RuntimeHelper::getReadFunction(mlir::PatternRewriter &rewriter) {
  if (M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(readFnName))
    return mlir::SymbolRefAttr::get(readFnName, context);

  auto llvmVoidTy = mlir::LLVM::LLVMType::getVoidTy(llvmDialect);
  auto llvmI32Ty = mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
  auto llvmFnType = mlir::LLVM::LLVMType::getFunctionTy(llvmVoidTy, {llvmI32Ty},
                                                        /*isVarArg=*/true);

  mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(M.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(M.getLoc(), readFnName, llvmFnType);
  return mlir::SymbolRefAttr::get(readFnName, context);
}

mlir::SymbolRefAttr
RuntimeHelper::getFileReadFunction(mlir::PatternRewriter &rewriter) {
  if (M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(fileReadFnName))
    return mlir::SymbolRefAttr::get(fileReadFnName, context);

  auto llvmI32Ty = mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
  auto llvmFnType = mlir::LLVM::LLVMType::getFunctionTy(
      llvmI32Ty, {llvmI32Ty, llvmI32Ty}, /*isVarArg=*/true);

  mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(M.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(M.getLoc(), fileReadFnName,
                                          llvmFnType);
  return mlir::SymbolRefAttr::get(fileReadFnName, context);
}

mlir::SymbolRefAttr
RuntimeHelper::getStoIAFunction(mlir::PatternRewriter &rewriter) {
  if (M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(stringToIntArrayFnName))
    return mlir::SymbolRefAttr::get(stringToIntArrayFnName, context);

  auto llvmVoidTy = mlir::LLVM::LLVMType::getVoidTy(llvmDialect);
  auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  auto I32Ptr = mlir::LLVM::LLVMType::getInt32Ty(llvmDialect).getPointerTo();
  auto llvmFnType = mlir::LLVM::LLVMType::getFunctionTy(
      llvmVoidTy, {I8Ptr, I32Ptr}, /*isvarArg*/ false);

  mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(M.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(M.getLoc(), stringToIntArrayFnName,
                                          llvmFnType);
  return mlir::SymbolRefAttr::get(stringToIntArrayFnName, context);
}

// TODO CLEANUP : Remove lot of repeated code!
mlir::SymbolRefAttr
RuntimeHelper::getStringToIntFunction(mlir::PatternRewriter &rewriter) {
  if (M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(stringToIntFnName))
    return mlir::SymbolRefAttr::get(stringToIntFnName, context);

  auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  auto I32 = mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
  auto llvmFnType =
      mlir::LLVM::LLVMType::getFunctionTy(I32, {I8Ptr}, /*isvarArg*/ false);

  mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(M.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(M.getLoc(), stringToIntFnName,
                                          llvmFnType);
  return mlir::SymbolRefAttr::get(stringToIntFnName, context);
}

mlir::SymbolRefAttr
RuntimeHelper::getStrCpyFunction(mlir::PatternRewriter &rewriter) {
  const char *fnName = "strcpy";

  if (M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(fnName))
    return mlir::SymbolRefAttr::get(fnName, context);

  auto llvmVoidTy = mlir::LLVM::LLVMType::getVoidTy(llvmDialect);
  auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  auto llvmFnType = mlir::LLVM::LLVMType::getFunctionTy(
      llvmVoidTy, {I8Ptr, I8Ptr}, /*isvarArg*/ false);

  mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(M.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(M.getLoc(), fnName, llvmFnType);
  return mlir::SymbolRefAttr::get(fnName, context);
}

mlir::SymbolRefAttr
RuntimeHelper::getStrCatFunction(mlir::PatternRewriter &rewriter) {
  const char *fnName = "strcat";

  if (M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(fnName))
    return mlir::SymbolRefAttr::get(fnName, context);

  auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  auto llvmFnType = mlir::LLVM::LLVMType::getFunctionTy(I8Ptr, {I8Ptr, I8Ptr},
                                                        /*isvarArg*/ false);

  mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(M.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(M.getLoc(), fnName, llvmFnType);
  return mlir::SymbolRefAttr::get(fnName, context);
}

mlir::SymbolRefAttr
RuntimeHelper::getTrimFunction(mlir::PatternRewriter &rewriter) {
  if (M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(trimFnName))
    return mlir::SymbolRefAttr::get(trimFnName, context);

  auto llvmVoidTy = mlir::LLVM::LLVMType::getVoidTy(llvmDialect);
  auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  auto I64 = mlir::LLVM::LLVMType::getInt64Ty(llvmDialect);
  auto llvmFnType =
      mlir::LLVM::LLVMType::getFunctionTy(llvmVoidTy, {I8Ptr, I64},
                                          /*isvarArg*/ false);

  mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(M.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(M.getLoc(), trimFnName, llvmFnType);
  return mlir::SymbolRefAttr::get(trimFnName, context);
}

mlir::SymbolRefAttr
RuntimeHelper::getIntToStringFunction(mlir::PatternRewriter &rewriter) {
  if (M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(intToStringFnName))
    return mlir::SymbolRefAttr::get(intToStringFnName, context);

  auto llvmVoidTy = mlir::LLVM::LLVMType::getVoidTy(llvmDialect);
  auto I8Ptr = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  auto I32 = mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
  auto llvmFnType = mlir::LLVM::LLVMType::getFunctionTy(
      llvmVoidTy, {I8Ptr, I32}, /*isvarArg*/ false);

  mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(M.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(M.getLoc(), intToStringFnName,
                                          llvmFnType);
  return mlir::SymbolRefAttr::get(intToStringFnName, context);
}

mlir::SymbolRefAttr
RuntimeHelper::getSprintfFunction(mlir::PatternRewriter &rewriter) {

  if (M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(sprintfFnName))
    return mlir::SymbolRefAttr::get(sprintfFnName, context);

  auto llvmI32Ty = mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
  auto llvmVoidTy = mlir::LLVM::LLVMType::getVoidTy(llvmDialect);
  auto llvmFnType = mlir::LLVM::LLVMType::getFunctionTy(llvmVoidTy, {llvmI32Ty},
                                                        /*isVarArg=*/true);

  mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(M.getBody());
  rewriter.create<mlir::LLVM::LLVMFuncOp>(M.getLoc(), sprintfFnName,
                                          llvmFnType);
  return mlir::SymbolRefAttr::get(sprintfFnName, context);
}
