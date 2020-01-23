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
#ifndef MLIR_FC_PASS_H
#define MLIR_FC_PASS_H

#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createForOpConverterPass();

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createFCDoConverterPass();

std::unique_ptr<mlir::Pass> createArrayOpsLoweringPass();

std::unique_ptr<mlir::Pass> createFCToLLVMLoweringPass();

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>> createOpenMPLoweringPass();

std::unique_ptr<mlir::Pass> createLoopStructureLoweringPass();

std::unique_ptr<mlir::Pass> createMemToRegPass();

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createSimplifyCFGPass();

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>>
createSimplifyLoopMemOperations();

std::unique_ptr<mlir::Pass> createNestedPUVariableLoweringPass();

std::unique_ptr<mlir::Pass> createProgramUnitLoweringPass();

#endif
