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
#include "AST/ProgramUnit.h"
#include "AST/SymbolTable.h"
#include "AST/Type.h"
#include "codegen/CGASTHelper.h"
#include "codegen/CodeGen.h"
#include "common/Debug.h"
#include "dialect/FCOps/FCOps.h"
#include "sema/Intrinsics.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

#include "AST/Expressions.h"

using namespace fc;
using namespace ast;

// TODO : Move to a helper class?
mlir::SymbolRefAttr CodeGen::getIntrinFunction(llvm::StringRef fnName,
                                               mlir::Type type) {
  auto string = fnName.str();
  if (type.isF32()) {
    string += ".f32";
  } else if (type.isF64()) {
    string += ".f64";
  } else {
    assert(false && "unknown intrinsic type!");
  }
  if (auto funcOp = theModule->lookupSymbol<FC::FCFuncOp>(string)) {
    assert(funcOp);
    return mlir::SymbolRefAttr::get(string, &mlirContext);
  }

  auto insertPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(theModule->getBody());

  llvm::ArrayRef<mlir::Type> tys{type};
  auto mlirFuncType = mlir::FunctionType::get(tys, tys, &mlirContext);
  auto mlirloc = builder.getUnknownLoc();
  builder.create<FC::FCFuncOp>(mlirloc, string, mlirFuncType);
  builder.restoreInsertionPoint(insertPt);
  return mlir::SymbolRefAttr::get(string, &mlirContext);
}

mlir::SymbolRefAttr CodeGen::getFmaxFunction() {
  auto fnName = "llvm.maxnum.f64";
  if (auto funcOp = theModule->lookupSymbol<FC::FCFuncOp>(fnName)) {
    assert(funcOp);
    return mlir::SymbolRefAttr::get(fnName, &mlirContext);
  }

  auto insertPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(theModule->getBody());
  auto F64 = builder.getF64Type();
  llvm::ArrayRef<mlir::Type> inputTys{F64, F64};
  llvm::ArrayRef<mlir::Type> resultTys{F64};
  auto mlirFuncType =
      mlir::FunctionType::get(inputTys, resultTys, &mlirContext);
  auto mlirloc = builder.getFileLineColLoc(
      builder.getIdentifier(this->parseTree->getName()), 1, 1);
  builder.create<FC::FCFuncOp>(mlirloc, fnName, mlirFuncType);
  builder.restoreInsertionPoint(insertPt);
  return mlir::SymbolRefAttr::get(fnName, &mlirContext);
}

mlir::Value CodeGen::emitCastExpr(mlir::Value fromVal, mlir::Type toType) {
  auto fromType = fromVal.getType();

  if (fromType == toType)
    return fromVal;

  auto mlirloc = fromVal.getLoc();
  if (toType.isa<mlir::IntegerType>() && fromType.isa<mlir::IndexType>()) {
    auto castOp = builder.create<mlir::IndexCastOp>(mlirloc, toType, fromVal);
    return castOp.getResult();
  } else if (fromType.isa<mlir::IntegerType>() &&
             toType.isa<mlir::IndexType>()) {
    auto castOp = builder.create<mlir::IndexCastOp>(mlirloc, toType, fromVal);
    return castOp.getResult();
  } else if (toType.isF32() && fromType.isF64()) {
    auto castOp = builder.create<mlir::FPTruncOp>(mlirloc, toType, fromVal);
    return castOp.getResult();
  } else if (toType.isF64() && fromType.isF32()) {
    auto castOp = builder.create<mlir::FPExtOp>(mlirloc, toType, fromVal);
    return castOp.getResult();
  } else if (toType.isa<mlir::IntegerType>() &&
             fromType.isa<mlir::FloatType>()) {
    auto castOp = builder.create<FC::CastOp>(mlirloc, toType, fromVal);
    return castOp.getResult();
  } else if (toType.isa<mlir::FloatType>() &&
             fromType.isa<mlir::IntegerType>()) {
    auto castOp = builder.create<mlir::SIToFPOp>(mlirloc, toType, fromVal);
    return castOp.getResult();
  } else if (toType.isa<mlir::IntegerType>() &&
             fromType.isa<mlir::IntegerType>()) {
    if (toType.cast<mlir::IntegerType>().getWidth() <
        fromType.cast<mlir::IntegerType>().getWidth()) {
      auto castOp = builder.create<mlir::TruncateIOp>(mlirloc, toType, fromVal);
      return castOp.getResult();
    } else if (toType.cast<mlir::IntegerType>().getWidth() >
               fromType.cast<mlir::IntegerType>().getWidth()) {
      auto castOp =
          builder.create<mlir::SignExtendIOp>(mlirloc, toType, fromVal);
      return castOp.getResult();
    } else {
      llvm_unreachable("Unhandled\n");
    }
  } else {
    auto castOp = builder.create<FC::CastOp>(mlirloc, toType, fromVal);
    return castOp.getResult();
  }
}

mlir::Value CodeGen::emitConstant(llvm::ArrayRef<llvm::StringRef> valueList,
                                  fc::Type *fcType, fc::Type *lhsTy,
                                  bool constructOnlyConstant) {

  llvm_unreachable("emitconstant");
}

mlir::Value CodeGen::castIntToFP(mlir::Value val, mlir::Type castType) {
  auto mlirloc = val.getLoc();
  auto castOp = builder.create<mlir::SIToFPOp>(mlirloc, castType, val);
  return castOp.getResult();
}

FC::FCFuncOp CodeGen::getOrInsertFuncOp(std::string name,
                                        llvm::ArrayRef<mlir::Type> argTys,
                                        mlir::Type retTy) {
  FC::FCFuncOp mlirFunc = nullptr;

  if (!(theModule->lookupSymbol(name))) {
    auto mlirFuncType =
        mlir::FunctionType::get(argTys, {retTy}, theModule->getContext());
    auto mlirloc = builder.getFileLineColLoc(
        builder.getIdentifier(this->parseTree->getName()), 1, 1);
    mlirFunc = FC::FCFuncOp::create(mlirloc, name, mlirFuncType);
    theModule->push_back(mlirFunc);
  } else {
    mlirFunc = llvm::dyn_cast<FC::FCFuncOp>(theModule->lookupSymbol(name));
  }

  assert(mlirFunc);
  return mlirFunc;
}

mlir::Value CodeGen::getMLIRBinaryOp(mlir::Value lhsVal, mlir::Value rhsVal,
                                     fc::ast::BinaryOpKind opKind) {

  bool isIndex = lhsVal.getType().isa<mlir::IndexType>();
  bool isRHSIndex = rhsVal.getType().isa<mlir::IndexType>();

  bool isInt = lhsVal.getType().isa<mlir::IntegerType>() || isIndex;

  bool isRHSInt = rhsVal.getType().isa<mlir::IntegerType>() || isRHSIndex;
  auto mlirloc = lhsVal.getLoc();
  assert(isInt == isRHSInt);

  // If one of them is index type. Convert both to index.
  if (isIndex != isRHSIndex) {
    if (isIndex) {
      assert(isRHSInt);
      rhsVal = builder.create<mlir::IndexCastOp>(mlirloc, rhsVal,
                                                 builder.getIndexType());
    } else {
      assert(isInt);
      lhsVal = builder.create<mlir::IndexCastOp>(mlirloc, lhsVal,
                                                 builder.getIndexType());
    }
  }

  switch (opKind) {
  case BinaryOpKind::Addition: {
    if (isInt)
      return builder.create<mlir::AddIOp>(mlirloc, lhsVal, rhsVal).getResult();
    return builder.create<mlir::AddFOp>(mlirloc, lhsVal, rhsVal).getResult();
  }
  case BinaryOpKind::Subtraction: {
    if (isInt)
      return builder.create<mlir::SubIOp>(mlirloc, lhsVal, rhsVal).getResult();
    return builder.create<mlir::SubFOp>(mlirloc, lhsVal, rhsVal).getResult();
  }
  case BinaryOpKind::Multiplication: {
    if (isInt)
      return builder.create<mlir::MulIOp>(mlirloc, lhsVal, rhsVal).getResult();
    return builder.create<mlir::MulFOp>(mlirloc, lhsVal, rhsVal).getResult();
  }
  case BinaryOpKind::Division: {
    if (isInt)
      return builder.create<mlir::SignedDivIOp>(mlirloc, lhsVal, rhsVal)
          .getResult();
    return builder.create<mlir::DivFOp>(mlirloc, lhsVal, rhsVal).getResult();
  }
  case BinaryOpKind::Mod: {
    if (isInt)
      return builder.create<mlir::SignedRemIOp>(mlirloc, lhsVal, rhsVal)
          .getResult();
    return builder.create<mlir::RemFOp>(mlirloc, lhsVal, rhsVal).getResult();
  }
  case BinaryOpKind::Concat: {
    return builder
        .create<FC::StrCatOp>(mlirloc, lhsVal.getType(), lhsVal, rhsVal)
        .getResult();
  }

  case BinaryOpKind::Power: {
    bool convertedRHS = false;
    bool convertedLHS = false;
    mlir::Type mlirF32Ty = mlir::FloatType::getF32(&mlirContext),
               mlirF64Ty = mlir::FloatType::getF64(&mlirContext),
               mlirI32Ty = mlir::IntegerType::get(32, &mlirContext);

    auto OrigType = lhsVal.getType();

    if (isInt) {
      lhsVal = castIntToFP(lhsVal, mlirF32Ty);
      convertedLHS = true;
    }

    if (isInt && isRHSInt) {
      rhsVal = castIntToFP(rhsVal, lhsVal.getType());
      convertedRHS = true;
    }

    // FIXME, we need (or to find?) this in the mlir repo
    auto isFloatTy = [](mlir::Type ty) {
      return ty.isF16() || ty.isF32() || ty.isF64();
    };

    if (lhsVal.getType() != rhsVal.getType() && isFloatTy(rhsVal.getType()) &&
        isFloatTy(lhsVal.getType())) {
      rhsVal = builder.create<FC::CastOp>(mlirloc, lhsVal.getType(), rhsVal)
                   .getResult();
      /* rhsVal = IRB->CreateFPCast(rhsVal, lhsVal.getType(), "fpcast"); */
    }

    assert(isFloatTy(lhsVal.getType()));

    FC::FCFuncOp powFunc = nullptr;
    if (lhsVal.getType().isF32() && rhsVal.getType().isF32()) {
      powFunc =
          getOrInsertFuncOp("llvm.pow.f32", {mlirF32Ty, mlirF32Ty}, mlirF32Ty);

    } else if (lhsVal.getType().isF64() && rhsVal.getType().isF64()) {
      powFunc =
          getOrInsertFuncOp("llvm.pow.f64", {mlirF64Ty, mlirF64Ty}, mlirF64Ty);

    } else {
      assert(rhsVal.getType().isInteger(32));
      if (lhsVal.getType().isF32()) {
        powFunc = getOrInsertFuncOp("llvm.powi.f32", {mlirF32Ty, mlirI32Ty},
                                    mlirF32Ty);

      } else if (lhsVal.getType().isF64()) {
        powFunc = getOrInsertFuncOp("llvm.powi.f64", {mlirF64Ty, mlirI32Ty},
                                    mlirF64Ty);
      }
    }

    assert(powFunc);

    llvm::SmallVector<mlir::Value, 2> funcArgList;
    funcArgList.push_back(lhsVal);
    funcArgList.push_back(rhsVal);

    mlir::Value finalVal = nullptr;
    finalVal = builder.create<FC::FCCallOp>(mlirloc, powFunc, funcArgList)
                   .getResult(0);

    if (convertedLHS && convertedRHS) {
      finalVal =
          builder.create<FC::CastOp>(mlirloc, OrigType, finalVal).getResult();
    }

    return finalVal;
  }
  default: { llvm_unreachable("unhandled binary expresssion"); }
  };
}

mlir::Value CodeGen::getMLIRArrayBinaryOp(mlir::Value lhsVal,
                                          mlir::Value rhsVal,
                                          fc::ast::BinaryOpKind opKind) {

  assert(lhsVal && rhsVal);

  auto lhsArrTy = lhsVal.getType().cast<FC::ArrayType>();
  auto rhsArrTy = rhsVal.getType().cast<FC::ArrayType>();
  auto isInteger = lhsArrTy.getEleTy().isIntOrIndex();
  auto isRHSInteger = rhsArrTy.getEleTy().isIntOrIndex();
  auto loc = lhsVal.getLoc();
  assert(isInteger == isRHSInteger);

  switch (opKind) {
  case BinaryOpKind::Addition: {
    if (isInteger)
      return builder.create<FC::ArrayAddIOp>(loc, lhsVal, rhsVal).getResult();
    return builder.create<FC::ArrayAddFOp>(loc, lhsVal, rhsVal).getResult();
  }
  case BinaryOpKind::Subtraction: {
    if (isInteger)
      return builder.create<FC::ArraySubIOp>(loc, lhsVal, rhsVal).getResult();
    return builder.create<FC::ArraySubFOp>(loc, lhsVal, rhsVal).getResult();
  }
  case BinaryOpKind::Multiplication: {
    if (isInteger)
      return builder.create<FC::ArrayMulIOp>(loc, lhsVal, rhsVal).getResult();
    return builder.create<FC::ArrayMulFOp>(loc, lhsVal, rhsVal).getResult();
  }
  case BinaryOpKind::Division: {
    if (isInteger)
      return builder.create<FC::ArrayDivIOp>(loc, lhsVal, rhsVal).getResult();
    return builder.create<FC::ArrayDivFOp>(loc, lhsVal, rhsVal).getResult();
  }
  case BinaryOpKind::Mod: {
    if (isInteger)
      return builder.create<FC::ArrayModIOp>(loc, lhsVal, rhsVal).getResult();
    return builder.create<FC::ArrayModFOp>(loc, lhsVal, rhsVal).getResult();
  }
  default: { llvm_unreachable("unhandled binary expresssion"); }
  };
}

// FIXME: This is not elementwise compare. Currently works for
// strcmp like usecases.
mlir::Value CodeGen::getMLIRArrayRelationalOp(mlir::Value lhsVal,
                                              mlir::Value rhsVal,
                                              RelationalOpKind opKind) {

  assert(lhsVal && rhsVal);

  auto lhsArrTy = lhsVal.getType().cast<FC::ArrayType>();
  auto rhsArrTy = rhsVal.getType().cast<FC::ArrayType>();
  auto isInteger = lhsArrTy.getEleTy().isIntOrIndex();
  auto isRHSInteger = rhsArrTy.getEleTy().isIntOrIndex();
  auto loc = lhsVal.getLoc();
  assert(isInteger == isRHSInteger);

  switch (opKind) {
  case RelationalOpKind::EQ:
    if (isInteger)
      return builder.create<FC::ArrayCmpIOp>(loc, mlir::CmpIPredicate::eq,
                                             lhsVal, rhsVal);
    return builder.create<FC::ArrayCmpFOp>(loc, mlir::CmpFPredicate::OEQ,
                                           lhsVal, rhsVal);
  case RelationalOpKind::NE:
    if (isInteger)
      return builder.create<FC::ArrayCmpIOp>(loc, mlir::CmpIPredicate::ne,
                                             lhsVal, rhsVal);
    return builder.create<FC::ArrayCmpFOp>(loc, mlir::CmpFPredicate::ONE,
                                           lhsVal, rhsVal);
  case RelationalOpKind::LT:
    if (isInteger)
      return builder.create<FC::ArrayCmpIOp>(loc, mlir::CmpIPredicate::slt,
                                             lhsVal, rhsVal);
    return builder.create<FC::ArrayCmpFOp>(loc, mlir::CmpFPredicate::OLT,
                                           lhsVal, rhsVal);
  case RelationalOpKind::LE:
    if (isInteger)
      return builder.create<FC::ArrayCmpIOp>(loc, mlir::CmpIPredicate::sle,
                                             lhsVal, rhsVal);
    return builder.create<FC::ArrayCmpFOp>(loc, mlir::CmpFPredicate::OLE,
                                           lhsVal, rhsVal);
  case RelationalOpKind::GT:
    if (isInteger)
      return builder.create<FC::ArrayCmpIOp>(loc, mlir::CmpIPredicate::sgt,
                                             lhsVal, rhsVal);
    return builder.create<FC::ArrayCmpFOp>(loc, mlir::CmpFPredicate::OGT,
                                           lhsVal, rhsVal);
  case RelationalOpKind::GE:
    if (isInteger)
      return builder.create<FC::ArrayCmpIOp>(loc, mlir::CmpIPredicate::sge,
                                             lhsVal, rhsVal);
    return builder.create<FC::ArrayCmpFOp>(loc, mlir::CmpFPredicate::OGE,
                                           lhsVal, rhsVal);
  default:
    llvm_unreachable("unhandled relational op");
  };
  return nullptr;
}

mlir::Value CodeGen::getMLIRRelationalOp(mlir::Value lhsVal, mlir::Value rhsVal,
                                         RelationalOpKind opKind) {

  assert(lhsVal && rhsVal);

  auto isInteger = lhsVal.getType().isIntOrIndex();
  auto isRHSInteger = rhsVal.getType().isIntOrIndex();
  auto loc = lhsVal.getLoc();
  assert(isInteger == isRHSInteger);
  if (isInteger && isRHSInteger) {
    if (lhsVal.getType().isIndex() != rhsVal.getType().isIndex())
      lhsVal = emitCastExpr(lhsVal, rhsVal.getType());
  }
  switch (opKind) {
  case RelationalOpKind::EQ:
    if (isInteger)
      return builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsVal,
                                          rhsVal);
    return builder.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OEQ, lhsVal,
                                        rhsVal);
  case RelationalOpKind::NE:
    if (isInteger)
      return builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ne, lhsVal,
                                          rhsVal);
    return builder.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::ONE, lhsVal,
                                        rhsVal);
  case RelationalOpKind::LT:
    if (isInteger)
      return builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, lhsVal,
                                          rhsVal);
    return builder.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OLT, lhsVal,
                                        rhsVal);
  case RelationalOpKind::LE:
    if (isInteger)
      return builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sle, lhsVal,
                                          rhsVal);
    return builder.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OLE, lhsVal,
                                        rhsVal);
  case RelationalOpKind::GT:
    if (isInteger)
      return builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, lhsVal,
                                          rhsVal);
    return builder.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OGT, lhsVal,
                                        rhsVal);
  case RelationalOpKind::GE:
    if (isInteger)
      return builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sge, lhsVal,
                                          rhsVal);
    return builder.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OGE, lhsVal,
                                        rhsVal);
  default:
    llvm_unreachable("unhandled llvm  relational op");
  };
  return nullptr;
}

mlir::Value CodeGen::getMLIRLogicalOp(mlir::Value lhsVal, mlir::Value rhsVal,
                                      LogicalOpKind opKind) {
  auto loc = rhsVal.getLoc();
  switch (opKind) {
  case LogicalOpKind::AND:
    return builder.create<mlir::AndOp>(loc, lhsVal, rhsVal);
  case LogicalOpKind::OR:
    return builder.create<mlir::OrOp>(loc, lhsVal, rhsVal);
  case LogicalOpKind::NEQV:
    return builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ne, lhsVal,
                                        rhsVal);
  case LogicalOpKind::EQV:
    return builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhsVal,
                                        rhsVal);
  case LogicalOpKind::NOT: {
    auto trueVal =
        builder.create<mlir::ConstantIntOp>(loc, 1, builder.getIntegerType(1));
    auto falseVal =
        builder.create<mlir::ConstantIntOp>(loc, 0, builder.getIntegerType(1));
    return builder.create<mlir::SelectOp>(loc, rhsVal, falseVal, trueVal);
  }
  default:
    llvm_unreachable("unhandled logical op");
  }
  return nullptr;
}

mlir::Value CodeGen::emitSizeForArrBounds(ArrayBounds &bounds) {
  mlir::Value lowerVal = nullptr;
  auto mlirloc = builder.getUnknownLoc();
  lowerVal = builder.create<mlir::ConstantIntOp>(mlirloc, bounds.first, 64);
  auto upperVal =
      builder.create<mlir::ConstantIntOp>(mlirloc, bounds.second, 64);

  // Size of the current dimension is {upper - lower + 1}
  auto sizeVal =
      builder.create<mlir::SubIOp>(mlirloc, upperVal, lowerVal).getResult();
  auto constOne = builder.create<mlir::ConstantIntOp>(mlirloc, 1, 64);
  sizeVal = builder.create<mlir::AddIOp>(mlirloc, sizeVal, constOne);
  return sizeVal;
}

mlir::Value CodeGen::emitDynArrayElement(ArrayElement *arrEle, bool isLHS,
                                         mlir::Value addr) {
  llvm_unreachable("Dynamic array element");
}

mlir::Value CodeGen::emitf77DynArrayElement(ArrayElement *arrEle, bool isLHS,
                                            mlir::Value addr) {
  llvm_unreachable("Dynamic array element");
}

mlir::Value CodeGen::expandIntrinsic(FunctionReference *funcReference) {

  auto sym = funcReference->getSymbol()->getOrigSymbol();
  auto loc = getLoc(funcReference->getSourceLoc());
  auto argList = funcReference->getArgsList();

  auto intrinKind = fc::intrin::getIntrinsicKind(sym->getName());
  switch (intrinKind) {
  case fc::intrin::mod: {
    assert(argList.size() == 2);
    return getMLIRBinaryOp(emitExpression(argList[0]),
                           emitExpression(argList[1]), BinaryOpKind::Mod);
  }
  case fc::intrin::command_argument_count: {
    assert(argList.size() == 0);
    return builder.create<FC::ArgcOp>(loc, builder.getIntegerType(32))
        .getResult();
  }
  case fc::intrin::exp: {
    assert(argList.size() == 1);
    auto Val = emitExpression(argList[0]);
    return builder.create<mlir::ExpOp>(loc, Val.getType(), Val);
  }
  case fc::intrin::sqrt: {
    assert(argList.size() == 1);
    auto arg = emitExpression(argList[0]);
    auto type = arg.getType();
    assert(type.isF32() || type.isF64());
    auto sqrtFn = getIntrinFunction("llvm.sqrt", type);
    auto callOp = builder.create<FC::FCCallOp>(
        arg.getLoc(), sqrtFn, llvm::ArrayRef<mlir::Type>{type},
        llvm::ArrayRef<mlir::Value>{arg});
    return callOp.getResult(0);
  }
  case fc::intrin::abs: {
    assert(argList.size() == 1);
    // Integer abs is lowered in sema
    auto val = emitExpression(argList[0]);
    auto type = val.getType();
    auto absFn = getIntrinFunction("llvm.fabs", type);
    auto callOp = builder.create<FC::FCCallOp>(
        val.getLoc(), absFn, llvm::ArrayRef<mlir::Type>{type},
        llvm::ArrayRef<mlir::Value>{val});
    return callOp.getResult(0);
  }
  case fc::intrin::max: {
    assert(argList.size() == 2);
    // Integer max is lowered in sema
    auto arg1 = emitExpression(argList[0]);
    auto arg2 = emitExpression(argList[1]);
    auto toType = arg1.getType();
    arg1 = emitCastExpr(arg1, builder.getF64Type());
    arg2 = emitCastExpr(arg2, builder.getF64Type());
    auto maxFn = getFmaxFunction();
    auto F64 = builder.getF64Type();
    auto callOp = builder.create<FC::FCCallOp>(
        arg1.getLoc(), maxFn, llvm::ArrayRef<mlir::Type>{F64},
        llvm::ArrayRef<mlir::Value>{arg1, arg2});
    return emitCastExpr(callOp.getResult(0), toType);
  }
  case fc::intrin::lbound: {
    assert(argList.size() == 2);

    if (std == Standard::f77) {
      assert(false);
    }
    mlir::Value dimension = emitExpression(argList[1]);
    dimension = castToIndex(dimension);
    auto LLValue = emitExpression(argList[0], true);
    return builder
        .create<FC::LBoundOp>(dimension.getLoc(), builder.getIndexType(),
                              LLValue, dimension)
        .getResult();
  }
  case fc::intrin::ubound: {
    assert(argList.size() == 2);

    if (std == Standard::f77) {
      assert(false);
    }
    mlir::Value dimension = emitExpression(argList[1]);
    dimension = castToIndex(dimension);
    auto LLValue = emitExpression(argList[0], true);
    return builder
        .create<FC::UBoundOp>(dimension.getLoc(), builder.getIndexType(),
                              LLValue, dimension)
        .getResult();
  }
  case fc::intrin::INT: {
    assert(argList.size() == 1);
    return builder.create<FC::CastOp>(loc, builder.getIntegerType(32),
                                      emitExpression(argList[0]));
  }
  case fc::intrin::iachar: {
    assert(argList.size() == 1);
    auto charValue = emitExpression(argList[0]);
    return builder.create<mlir::SignExtendIOp>(loc, charValue,
                                               builder.getIntegerType(32));
  }
  case fc::intrin::real: {
    assert(argList.size() == 1);
    auto F32 = mlir::FloatType::get(mlir::StandardTypes::F32, &mlirContext);
    auto inputVal = emitExpression(argList[0]);
    if (inputVal.getType().isF64()) {
      return builder.create<mlir::FPTruncOp>(loc, F32, inputVal);
    }
    if (inputVal.getType().isF32()) {
      return inputVal;
    }
    if (inputVal.getType().isIntOrIndex()) {
      return builder.create<mlir::SIToFPOp>(loc, inputVal, F32);
    }
    assert(false && "unknown type for real() intrinsic");
  }
  case fc::intrin::cos: {
    assert(argList.size() == 1);
    auto arg = emitExpression(argList[0]);
    auto argType = arg.getType();
    assert(argType.isF32() || argType.isF64());
    auto cosFn = getIntrinFunction("llvm.cos", argType);
    auto callOp = builder.create<FC::FCCallOp>(
        arg.getLoc(), cosFn, llvm::ArrayRef<mlir::Type>{argType},
        llvm::ArrayRef<mlir::Value>{arg});
    return callOp.getResult(0);
    break;
  }
  case fc::intrin::sin: {
    assert(argList.size() == 1);
    auto arg = emitExpression(argList[0]);
    auto argType = arg.getType();
    assert(argType.isF32() || argType.isF64());
    auto sinFn = getIntrinFunction("llvm.sin", argType);
    auto callOp = builder.create<FC::FCCallOp>(
        arg.getLoc(), sinFn, llvm::ArrayRef<mlir::Type>{argType},
        llvm::ArrayRef<mlir::Value>{arg});
    return callOp.getResult(0);
    break;
  }
  case fc::intrin::log: {
    assert(argList.size() == 1);
    auto arg = emitExpression(argList[0]);
    auto argType = arg.getType();
    assert(argType.isF32() || argType.isF64());
    auto logFn = getIntrinFunction("llvm.log", argType);
    auto callOp = builder.create<FC::FCCallOp>(
        arg.getLoc(), logFn, llvm::ArrayRef<mlir::Type>{argType},
        llvm::ArrayRef<mlir::Value>{arg});
    return callOp.getResult(0);
    break;
  }
  default:
    return nullptr;
  };
}

mlir::Value CodeGen::emitStaticArrayElement(ArrayElement *arrEle, bool isLHS,
                                            mlir::Value addr) {
  llvm_unreachable("static array element");
}

mlir::Value CodeGen::emitFCArrayElement(ArrayElement *arrEle) {
  auto mlirloc = getLoc(arrEle->getSourceLoc());
  auto Alloca = context.symbolMap[arrEle->getSymbol()->getName()];
  assert(Alloca);
  llvm::SmallVector<mlir::Value, 2> subs;

  auto one = builder.create<mlir::ConstantIndexOp>(mlirloc, 0);
  assert(arrEle->getSubscriptList().size() == 1);
  for (auto sub : arrEle->getSubscriptList()) {
    auto tempSub = emitExpression(sub);
    if (!tempSub.getType().isa<mlir::IndexType>()) {
      tempSub = builder.create<mlir::IndexCastOp>(mlirloc, tempSub,
                                                  builder.getIndexType());
    }
    subs.push_back(tempSub);
  }

  auto arrayTy = llvm::dyn_cast<fc::ArrayType>(arrEle->getType());
  assert(arrayTy);
  bool isPartial = false;
  for (unsigned i = subs.size(); i < arrayTy->getNumDims(); ++i) {
    // TODO Push lbound
    subs.push_back(one);
    isPartial = true;
  }

  if (arrayTy->isStringArrTy() && isPartial) {
    std::reverse(subs.begin(), subs.end());
  }

  auto retType =
      FC::RefType::get(cgHelper->getMLIRTypeFor(arrEle->getElementType()));
  auto arrayOp = builder.create<FC::ArrayEleOp>(mlirloc, retType, Alloca, subs);
  return arrayOp.getResult();
}

mlir::Value CodeGen::emitArrayElement(ArrayElement *arrEle, bool isLHS,
                                      mlir::Value addr) {
  auto mlirloc = getLoc(arrEle->getSourceLoc());
  auto Alloca = context.symbolMap[arrEle->getSymbol()->getName()];
  assert(Alloca);
  FC::SubscriptRangeList subs;

  auto fcTy = arrEle->getSymbol()->getOrigType();
  auto fcArrayTy = llvm::dyn_cast<fc::ArrayType>(fcTy);
  assert(fcArrayTy);
  bool isString = false;
  if (fcArrayTy->getElementTy()->isStringCharTy()) {
    isString = true;
  }

  bool isPartial = false;
  if (arrEle->getSubscriptList().size() < fcArrayTy->getNumDims())
    isPartial = true;

  auto one = builder.create<mlir::ConstantIntOp>(mlirloc, 1, 32);
  for (auto sub : arrEle->getSubscriptList()) {
    auto tempSub = emitExpression(sub);
    if (isString && !isPartial)
      tempSub = builder.create<mlir::SubIOp>(
          mlirloc, tempSub, emitCastExpr(one, tempSub.getType()));
    subs.push_back(FC::SubscriptRange(castToIndex(tempSub)));
  }

  one = builder.create<mlir::ConstantIntOp>(mlirloc, 0, 32);
  for (unsigned i = subs.size(); i < fcArrayTy->getNumDims(); ++i) {
    // TODO Push lbound
    subs.push_back(FC::SubscriptRange(castToIndex(one)));
    isPartial = true;
  }

  if (fcArrayTy->isStringArrTy() && isPartial) {
    std::reverse(subs.begin(), subs.end());
  }

  auto loadOp = builder.create<FC::FCLoadOp>(mlirloc, Alloca, subs);
  loadOp.setAttr("name", builder.getStringAttr(arrEle->getSymbol()->getName()));
  return loadOp;
}

mlir::Value CodeGen::emitExpression(Expr *expr, bool isLHS) {
  auto kind = expr->getStmtType();
  auto mlirloc = getLoc(expr->getSourceLoc());
  switch (kind) {
  case StmtType::ConstantValKind: {
    auto Const = static_cast<ConstantVal *>(expr);
    auto type = Const->getType();
    if (type->isIntegralTy()) {
      auto constant = builder.create<mlir::ConstantIntOp>(
          mlirloc, Const->getInt(), type->getSizeInBits());
      return constant.getResult();
    }

    if (type->isRealTy() || type->isDoubleTy()) {
      auto str = Const->getValue();
      bool isDouble = type->isDoubleTy();
      for (unsigned k = 0; k < str.size(); ++k) {
        if (str[k] == 'd' || str[k] == 'D') {
          isDouble = true;
          str[k] = 'E';
        }
      }
      auto floatTy = isDouble ? mlir::FloatType::getF64(&mlirContext)
                              : mlir::FloatType::getF32(&mlirContext);

      if (isDouble) {
        auto floatVal = llvm::APFloat(::atof(str.c_str()));
        auto constant =
            builder.create<mlir::ConstantFloatOp>(mlirloc, floatVal, floatTy);
        return constant.getResult();
      }

      auto floatVal = llvm::APFloat((float)Const->getFloat());
      auto constant =
          builder.create<mlir::ConstantFloatOp>(mlirloc, floatVal, floatTy);
      return constant.getResult();
    }
    /*
    if (type->isRealTy()) {
      auto floatVal = llvm::APFloat((float)Const->getFloat());
      auto constant = builder.create<mlir::ConstantFloatOp>(
          mlirloc, floatVal, mlir::FloatType::getF32(&mlirContext));
      return constant.getResult();
    }
    if (type->isDoubleTy()) {
      auto floatVal = llvm::APFloat(Const->getFloat());
      auto constant = builder.create<mlir::ConstantFloatOp>(
          mlirloc, floatVal, mlir::FloatType::getF64(&mlirContext));
      return constant.getResult();
    }
    */

    auto ArrTy = llvm::dyn_cast<ArrayType>(type);
    if ((ArrTy && ArrTy->getElementTy()->isStringCharTy()) ||
        type->isCharacterTy()) {

      auto string = Const->getValueRef().str();
      string += '\0';
      long stringSize = (long)string.size();

      FC::ArrayType::Shape shape;
      shape.push_back({0, stringSize - 1, stringSize});
      auto stringType = FC::ArrayType::get(shape, builder.getIntegerType(8));
      auto stringVal = builder.create<FC::StringOp>(
          mlirloc, stringType, builder.getStringAttr(string));
      return stringVal.getResult();
    }

    if (type->isLogicalTy()) {
      int val = Const->getValueRef().equals_lower(".false.") ? 0 : 1;
      auto constant = builder.create<mlir::ConstantIntOp>(mlirloc, val, 1);
      return constant.getResult();
    }

    llvm_unreachable("unknown constant type");
  }

  case StmtType::ObjectNameKind: {
    auto objName = static_cast<ObjectName *>(expr);
    auto Alloca = context.getMLIRValueFor(objName->getName());

    assert(Alloca);
    if (isLHS || objName->getType()->isArrayTy() ||
        (Alloca.isa<mlir::BlockArgument>() &&
         Alloca.getType().isa<mlir::IndexType>()))
      return Alloca;
    return emitLoadInstruction(Alloca, objName->getName());
  }

  // Handle array access!
  case StmtType::ArrayElementKind: {
    auto arrEle = llvm::cast<fc::ArrayElement>(expr);
    if (!isLHS)
      return emitArrayElement(arrEle, isLHS);
    else
      return emitFCArrayElement(arrEle);
  }

  case StmtType::BinaryExprKind: {
    auto binaryExpr = static_cast<BinaryExpr *>(expr);
    auto lhs = binaryExpr->getLHS();
    auto rhs = binaryExpr->getRHS();
    auto OpKind = binaryExpr->getOpKind();
    if (!lhs) { // unary minus (note that unary plus is merely ignored in the
      // parser)
      assert(false);
      auto rhsVal = emitExpression(rhs);
      assert(rhsVal);
      assert(OpKind == BinaryOpKind::Subtraction);
      return builder
          .create<mlir::SubIOp>(
              mlirloc, builder.create<mlir::ConstantIntOp>(mlirloc, 0, 32),
              rhsVal)
          .getResult();
    }

    auto lhsVal = emitExpression(lhs);
    auto rhsVal = emitExpression(rhs);
    assert(lhsVal && rhsVal);

    // Concat operates on char arrays, but isn't translated to an MLIR array op.
    if (lhsVal.getType().isa<FC::ArrayType>() && OpKind != BinaryOpKind::Concat)
      return getMLIRArrayBinaryOp(lhsVal, rhsVal, OpKind);

    return getMLIRBinaryOp(lhsVal, rhsVal, OpKind);
  }
  case StmtType::RelationalExprKind: {
    auto relExpr = static_cast<RelationalExpr *>(expr);
    auto lhs = relExpr->getLHS();
    auto rhs = relExpr->getRHS();
    auto OpKind = relExpr->getOpKind();

    auto lhsVal = emitExpression(lhs);
    auto rhsVal = emitExpression(rhs);
    assert(lhsVal && rhsVal);

    if (lhsVal.getType().isa<FC::ArrayType>())
      return getMLIRArrayRelationalOp(lhsVal, rhsVal, OpKind);
    return getMLIRRelationalOp(lhsVal, rhsVal, OpKind);
  }
  case StmtType::LogicalExprKind: {
    auto logicalExpr = static_cast<LogicalExpr *>(expr);
    auto lhs = logicalExpr->getLHS();
    auto rhs = logicalExpr->getRHS();
    auto OpKind = logicalExpr->getOpKind();

    if (!lhs) { // no lhs for .NOT.
      auto rhsVal = emitExpression(rhs);
      assert(rhsVal);
      return getMLIRLogicalOp(nullptr, rhsVal, OpKind);
    }
    auto lhsVal = emitExpression(lhs);
    auto rhsVal = emitExpression(rhs);
    assert(lhsVal && rhsVal);
    return getMLIRLogicalOp(lhsVal, rhsVal, OpKind);
  }
  case StmtType::CastExprKind: {
    auto castExpr = static_cast<CastExpr *>(expr);
    auto fromExpr = castExpr->getExpr();
    auto fromVal = emitExpression(fromExpr);
    return emitCastExpr(fromVal, cgHelper->getMLIRTypeFor(castExpr->getType()));
  }

  case StmtType::FunctionReferenceKind: {

    auto funcReference = static_cast<FunctionReference *>(expr);
    if (auto val = expandIntrinsic(funcReference)) {
      return val;
    }
    auto sym = funcReference->getSymbol()->getOrigSymbol();
    auto argList = funcReference->getArgsList();
    auto Call = emitCall(sym, argList);
    return Call->getResult(0);
  }

  case StmtType::ArraySectionKind: {
    auto arrSec = static_cast<ArraySection *>(expr);
    assert(arrSec->isFullRange());
    auto Alloca = context.getMLIRValueFor(arrSec->getName());
    assert(Alloca);
    if (isLHS)
      return Alloca;
    return emitLoadInstruction(Alloca, arrSec->getName());
  }
  default:
    llvm_unreachable("Unkown base expression type.");
  };
  return nullptr;
}

mlir::Value CodeGen::getArrDimSizeVal(Expr *expr, mlir::Value exprVal) {
  auto arrEle = llvm::dyn_cast<ArraySection>(expr);
  if (!arrEle)
    return nullptr;

  Type *arrEleType = arrEle->getType();

  if (auto arrElePtrTy = llvm::dyn_cast<fc::PointerType>(arrEleType)) {
    arrEleType = arrElePtrTy->getElementType();
  }

  if (!arrEleType->isArrayTy())
    return nullptr;
  auto arrType = static_cast<ArrayType *>(arrEleType);

  // If this is a static array.
  if (!arrType->boundsEmpty()) {
    auto boundsList = arrType->getBoundsList();

    mlir::Value sizeVal = nullptr;
    for (unsigned i = 0; i < boundsList.size(); ++i) {
      auto currSizeVal = emitSizeForArrBounds(boundsList[i]);
      if (!sizeVal) {
        sizeVal = currSizeVal;
        continue;
      }
      sizeVal = builder.create<mlir::MulIOp>(getLoc(expr->getSourceLoc()),
                                             sizeVal, currSizeVal);
    }
    return sizeVal;
  }
  llvm_unreachable("Dynamic array size");
}
