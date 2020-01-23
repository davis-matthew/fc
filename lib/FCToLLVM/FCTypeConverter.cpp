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
#include "FCToLLVM/FCToLLVMLowering.h"
#include "dialect/FC/FCOps.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"

#define PASS_NAME "FCToLLVMLowering"
#define DEBUG_TYPE PASS_NAME
// TODO: Add DEBUG WITH TYPE and STATISTIC

using namespace fcmlir;
using namespace mlir;
using namespace FC;

LLVM::LLVMType FCTypeConverter::convertArrayType(FC::ArrayType type) {
  LLVM::LLVMType elementType =
      convertType(type.getEleTy()).cast<LLVM::LLVMType>();
  if (!elementType)
    return {};
  auto ptrTy = elementType.getPointerTo();
  auto mlirIndexTy = mlir::IndexType::get(type.getContext());
  auto indexTy = convertType(mlirIndexTy).cast<LLVM::LLVMType>();
  auto arrayTy = LLVM::LLVMType::getArrayTy(indexTy, type.getRank());
  return LLVM::LLVMType::getStructTy(ptrTy, indexTy, arrayTy, arrayTy, arrayTy,
                                     arrayTy);
}

Type FCTypeConverter::convertType(Type t) {
  if (auto complex = t.dyn_cast<mlir::ComplexType>()) {
    auto eleTy = convertType(complex.getElementType()).cast<LLVM::LLVMType>();
    return LLVM::LLVMType::getArrayTy(eleTy, 2);
    return LLVM::LLVMType::getStructTy(eleTy, eleTy);
  }
  switch (t.getKind()) {
  case FC::FC_Ref: {
    auto eleTy = t.cast<FC::RefType>().getEleTy();
    if (auto arrTy = eleTy.dyn_cast<FC::ArrayType>()) {
      return convertArrayType(arrTy);
    }
    bool isDoublePtr = false;
    if (auto refType = eleTy.dyn_cast<FC::RefType>()) {
      eleTy = refType.getEleTy();
      isDoublePtr = true;
    }
    auto Ty = FCTypeConverter::convertType(eleTy);
    assert(Ty && "Failed to convert from std to llvm type");
    auto llTy = Ty.cast<LLVM::LLVMType>();
    if (isDoublePtr) {
      return llTy.getPointerTo().getPointerTo();
    }
    return llTy.getPointerTo();
  } break;
  case FC::FC_Pointer: {
    auto eleTy = t.cast<FC::PointerType>().getEleTy();
    auto Ty = FCTypeConverter::convertType(eleTy).cast<LLVM::LLVMType>();
    return Ty.getPointerTo();
  } break;
  case FC::FC_Array: {
    auto arrTy = t.cast<FC::ArrayType>();
    return convertArrayType(arrTy);
  } break;
  default:
    return LLVMTypeConverter::convertType(t);
  }
}
