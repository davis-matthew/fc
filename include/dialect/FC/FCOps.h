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
//===- FCOps.h FC specific Ops ----------------------------------===//

#ifndef MLIR_DIALECT_FORTRANOPS_FORTRANOPS_H
#define MLIR_DIALECT_FORTRANOPS_FORTRANOPS_H

#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/LoopLikeInterface.h"
#include "llvm/ADT/BitVector.h"

using namespace mlir;

namespace FC {

namespace detail {
struct ArrayTypeStorage;
struct RefTypeStorage;
struct SubscriptRangeAttrStorage;
struct PointerTypeStorage;
} // namespace detail

// A data structure to hold the information about the
// array sections. IT is currently used in fc.load
// and store operations.
class SubscriptRange {
private:
  mlir::Value lowerBoundVal{nullptr};
  mlir::Value upperBoundVal{nullptr};
  mlir::Value strideVal{nullptr};
  bool hasRange{false};

public:
  explicit SubscriptRange(mlir::Value subscript)
      : lowerBoundVal(subscript), hasRange(false) {}
  explicit SubscriptRange(mlir::Value lb, mlir::Value ub, mlir::Value stride)
      : lowerBoundVal(lb), upperBoundVal(ub), strideVal(stride),
        hasRange(true) {}

  inline mlir::Value lowerBound() const {
    assert(hasRange);
    return lowerBoundVal;
  }

  inline bool isRangeType() const { return hasRange; }

  inline mlir::Value upperBound() const {
    assert(hasRange);
    return upperBoundVal;
  }

  inline mlir::Value stride() const {
    assert(hasRange);
    return strideVal;
  }

  // Utility function for array element.
  inline mlir::Value subscript() const {
    assert(!hasRange);
    return lowerBoundVal;
  }
};

using SubscriptRangeList = llvm::SmallVector<SubscriptRange, 2>;

class FCOpsDialect : public mlir::Dialect {
public:
  FCOpsDialect(mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "fc"; }

  void printType(mlir::Type ty, mlir::DialectAsmPrinter &p) const;

  void printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter &p) const;
};

enum FCTypeKind {
  FC_Array = mlir::Type::FIRST_FC_TYPE,
  FC_Ref,
  FC_Pointer,
};

enum FCAttrKind {
  FC_Attr = mlir::Attribute::FIRST_FC_ATTR,
  FC_SubscriptRangeAttr,
  FC_StringInfoAttr,
};

class SubscriptRangeAttr
    : public mlir::Attribute::AttrBase<SubscriptRangeAttr, mlir::Attribute,
                                       detail::SubscriptRangeAttrStorage> {
public:
  using Base::Base;

  static llvm::StringRef getAttrName() { return "subscript_range"; }
  static SubscriptRangeAttr get(mlir::MLIRContext *ctxt,
                                llvm::BitVector indices = {});
  constexpr static bool kindof(unsigned kind) {
    return kind == FCAttrKind::FC_SubscriptRangeAttr;
  }

  llvm::BitVector getSubscriptRangeInfo() const;
};

// Used to differeniate between the string and char array.
class StringInfoAttr
    : public mlir::Attribute::AttrBase<StringInfoAttr, mlir::Attribute,
                                       detail::SubscriptRangeAttrStorage> {
public:
  using Base::Base;

  static llvm::StringRef getAttrName() { return "is_string"; }
  static StringInfoAttr get(mlir::MLIRContext *ctxt,
                            llvm::BitVector &stringInfo);
  constexpr static bool kindof(unsigned kind) {
    return kind == FCAttrKind::FC_StringInfoAttr;
  }

  llvm::BitVector getStringInfo() const;
};

class ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type,
                                              FC::detail::ArrayTypeStorage> {
public:
  static constexpr int64_t dynamicSizeValue = -1;
  static int64_t getDynamicSizeValue() { return dynamicSizeValue; }
  using Base::Base;
  // Contains lower bound, upper bound and size attribute.
  struct Dim {
    int64_t lowerBound, upperBound, size;
    Dim()
        : lowerBound(dynamicSizeValue), upperBound(dynamicSizeValue),
          size(dynamicSizeValue) {}
    static Dim getUnknown() { return Dim(); }
    Dim(int64_t lb, int64_t ub, int64_t size)
        : lowerBound(lb), upperBound(ub), size(size) {}
  };
  using Shape = llvm::SmallVector<Dim, 4>;

  mlir::Type getEleTy() const;

  Shape getShape() const;

  bool hasStaticShape() const {
    for (auto &dim : getShape()) {
      if (dim.size == ArrayType::getDynamicSizeValue()) {
        return false;
      }
    }
    return true;
  }
  unsigned getRank() const { return getShape().size(); }

  static ArrayType get(const Shape &shape, mlir::Type elementType);
  static bool kindof(unsigned kind) { return kind == FCTypeKind::FC_Array; }
};

bool operator==(const ArrayType::Shape &, const ArrayType::Shape &);
llvm::hash_code hash_value(const ArrayType::Shape &);

class RefType : public mlir::Type::TypeBase<RefType, mlir::Type,
                                            FC::detail::RefTypeStorage> {
public:
  using Base::Base;
  mlir::Type getEleTy() const;
  static bool kindof(unsigned kind) { return kind == FCTypeKind::FC_Ref; }
  static RefType get(mlir::Type elementType);

  bool isStatic() {
    auto eleTy = getEleTy();
    if (auto arrTy = eleTy.dyn_cast<FC::ArrayType>()) {
      return arrTy.hasStaticShape();
    }
    assert(eleTy.isIntOrIndexOrFloat());
    return true;
  }

  // This looks inside the ArrayType (if exists) and return the ele type.
  mlir::Type getUnderlyingEleType() {
    auto eleTy = getEleTy();
    if (auto arrTy = eleTy.dyn_cast<FC::ArrayType>()) {
      return arrTy.getEleTy();
    }
    assert(eleTy.isIntOrIndexOrFloat());
    return eleTy;
  }
};

// Represents fortran pointer type.
class PointerType
    : public mlir::Type::TypeBase<PointerType, mlir::Type,
                                  FC::detail::PointerTypeStorage> {
public:
  using Base::Base;
  mlir::Type getEleTy() const;
  static bool kindof(unsigned kind) { return kind == FCTypeKind::FC_Pointer; }
  static PointerType get(mlir::Type elementType);
}; // namespace FC

#define GET_OP_CLASSES
#include "dialect/FC/FCOps.h.inc"
} // namespace FC
#endif
