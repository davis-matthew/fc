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
#ifndef FC_CODEGEN_RUNTIME_HELPER_H
#define FC_CODEGEN_RUNTIME_HELPER_H

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/Transforms/DialectConversion.h"

namespace fc {
struct RuntimeHelper {
private:
  mlir::ModuleOp M;
  mlir::MLIRContext *context;
  mlir::LLVM::LLVMDialect *llvmDialect;

public:
  explicit RuntimeHelper(mlir::ModuleOp _M, mlir::MLIRContext *C,
                         mlir::LLVM::LLVMDialect *dialect)
      : M(_M), context(C), llvmDialect(dialect) {}

  // Names.
  const char *sprintfFnName = "__fc_runtime_sprintf";
  const char *printFnName = "__fc_runtime_print";
  const char *writeFnName = "__fc_runtime_write";
  const char *readFnName = "__fc_runtime_scan";
  const char *openFnName = "__fc_runtime_open";
  const char *fileReadFnName = "__fc_runtime_fread";
  const char *fileWriteFnName = "__fc_runtime_fwrite";
  const char *fileCloseFnName = "__fc_runtime_close";
  const char *stringToIntFnName = "__fc_runtime_stoi";
  const char *stringToIntArrayFnName = "__fc_runtime_stoia";
  const char *intToStringFnName = "__fc_runtime_itos";
  const char *isysClockFnName = "__fc_runtime_isysClock";
  const char *trimFnName = "__fc_runtime_trim";

  mlir::SymbolRefAttr getPrintFunction(mlir::PatternRewriter &rewriter);
  mlir::SymbolRefAttr getWriteFunction(mlir::PatternRewriter &rewriter);
  mlir::SymbolRefAttr getFileWriteFunction(mlir::PatternRewriter &rewriter);
  mlir::SymbolRefAttr getReadFunction(mlir::PatternRewriter &rewriter);
  mlir::SymbolRefAttr getFileReadFunction(mlir::PatternRewriter &rewriter);
  mlir::SymbolRefAttr getOpenFunction(mlir::PatternRewriter &rewriter);
  mlir::SymbolRefAttr getCloseFunction(mlir::PatternRewriter &rewriter);
  mlir::SymbolRefAttr getStoIAFunction(mlir::PatternRewriter &rewriter);
  mlir::SymbolRefAttr getStrCpyFunction(mlir::PatternRewriter &rewriter);
  mlir::SymbolRefAttr getIntToStringFunction(mlir::PatternRewriter &rewriter);
  mlir::SymbolRefAttr getSprintfFunction(mlir::PatternRewriter &rewriter);
  mlir::SymbolRefAttr getStringToIntFunction(mlir::PatternRewriter &rewriter);
  mlir::SymbolRefAttr getStrCatFunction(mlir::PatternRewriter &rewriter);
  mlir::SymbolRefAttr getTrimFunction(mlir::PatternRewriter &rewriter);
#if 0
  mlir::SymbolRefAttr getStrCmpFunction();
  mlir::SymbolRefAttr getISysClockFunction();
#endif

  void fillPrintArgsFor(mlir::Value val,
                        mlir::SmallVectorImpl<mlir::Value> &argsList,
                        mlir::Location loc, mlir::Value arrDimSize,
                        mlir::Type baseType,
                        mlir::ConversionPatternRewriter *builder,
                        bool isDynArr = false, bool isString = false);

  void fillReadArgsFor(mlir::Value val,
                       llvm::SmallVectorImpl<mlir::Value> &argsList,
                       mlir::Location loc, mlir::Value arrDimSize,
                       mlir::Type baseType,
                       mlir::ConversionPatternRewriter *IRB,
                       bool isString = false);
#if 0
  void fillOpenArgsFor(mlir::Value unit, mlir::Value fileName,
                       llvm::SmallVectorImpl<mlir::Value > &argList,
#endif
};
} // namespace fc

#endif
