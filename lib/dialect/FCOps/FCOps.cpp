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
//===- FCOps.cpp - FC Operations ---------------------------------===//

#include "dialect/FCOps/FCOps.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace FC;
using llvm::dbgs;

#define DEBUG_TYPE "fortran-ops"

//===----------------------------------------------------------------------===//
// FCOpsDialect
//===----------------------------------------------------------------------===//

FCOpsDialect::FCOpsDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<ArrayType, RefType>();
  addAttributes<SubscriptRangeAttr, StringInfoAttr>();
  addOperations<
#define GET_OP_LIST
#include "dialect/FCOps/FCOps.cpp.inc"
      >();
}

// compare if two shapes are equivalent
bool FC::operator==(const ArrayType::Shape &shape1,
                    const ArrayType::Shape &shape2) {
  if (shape1.size() != shape2.size())
    return false;
  for (std::size_t i = 0, e = shape1.size(); i != e; ++i)
    if (shape1[i].lowerBound != shape2[i].lowerBound ||
        shape1[i].upperBound != shape2[i].upperBound ||
        shape1[i].size != shape2[i].size)
      return false;
  return true;
}

// compute the hash of a Shape
llvm::hash_code FC::hash_value(const ArrayType::Shape &sh) {
  if (sh.empty()) {
    return llvm::hash_combine(0);
  }
  llvm::SmallVector<llvm::hash_code, 2> values(sh.size());
  for (auto &dim : sh) {
    llvm::SmallVector<llvm::hash_code, 4> dimValues;
    dimValues.push_back(dim.lowerBound);
    dimValues.push_back(dim.upperBound);
    // NOTE: size is anyway derived from lower and upper bounds.
    // but pushing it anyway!
    dimValues.push_back(dim.size);
    values.push_back(
        llvm::hash_combine_range(dimValues.begin(), dimValues.end()));
  }

  return llvm::hash_combine_range(values.begin(), values.end());
}

namespace FC {
namespace detail {
struct ArrayTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::tuple<ArrayType::Shape, mlir::Type>;

  static unsigned hashKey(const KeyTy &key) {
    auto shapeHash{FC::hash_value(std::get<ArrayType::Shape>(key))};
    return llvm::hash_combine(shapeHash, std::get<mlir::Type>(key));
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy{getShape(), getElementType()};
  }

  static ArrayTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    auto *storage = allocator.allocate<ArrayTypeStorage>();
    return new (storage) ArrayTypeStorage{std::get<ArrayType::Shape>(key),
                                          std::get<mlir::Type>(key)};
  }

  ArrayType::Shape getShape() const { return shape; }

  mlir::Type getElementType() const { return eleTy; }

protected:
  ArrayType::Shape shape;
  mlir::Type eleTy;

private:
  ArrayTypeStorage() = delete;
  explicit ArrayTypeStorage(const ArrayType::Shape &shape, mlir::Type eleTy)
      : shape{shape}, eleTy{eleTy} {}
};

struct RefTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static RefTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    auto *storage = allocator.allocate<RefTypeStorage>();
    return new (storage) RefTypeStorage{key};
  }

  mlir::Type getElementType() const { return eleTy; }

protected:
  mlir::Type eleTy;

private:
  RefTypeStorage() = delete;
  explicit RefTypeStorage(mlir::Type eleTy) : eleTy{eleTy} {}
};

/// An attribute representing a reference to a type.
struct SubscriptRangeAttrStorage : public mlir::AttributeStorage {
  using KeyTy = llvm::BitVector;

  explicit SubscriptRangeAttrStorage(KeyTy &_value) : value(_value) {}

  static unsigned hashKey(const KeyTy &key) {
    if (key.empty()) {
      return llvm::hash_combine(0);
    }
    llvm::SmallVector<llvm::hash_code, 2> values(key.size());
    for (unsigned I = 0; I < key.size(); ++I) {
      values[I] = llvm::hash_value(key[I]);
    }
    return llvm::hash_combine_range(values.begin(), values.end());
  }

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == value; }

  KeyTy get() { return value; }

  /// Construct a new storage instance.
  static SubscriptRangeAttrStorage *
  construct(mlir::AttributeStorageAllocator &allocator, KeyTy &key) {
    return new (allocator.allocate<SubscriptRangeAttrStorage>())
        SubscriptRangeAttrStorage(key);
  }

  KeyTy value;
};

} // namespace detail
} // namespace FC

ArrayType ArrayType::get(const Shape &shape, mlir::Type elementType) {
  auto *ctxt = elementType.getContext();
  return Base::get(ctxt, FC_Array, shape, elementType);
}

RefType RefType::get(mlir::Type elementType) {
  auto *ctxt = elementType.getContext();
  return Base::get(ctxt, FC_Ref, elementType);
}

mlir::Type ArrayType::getEleTy() const { return getImpl()->getElementType(); }

ArrayType::Shape ArrayType::getShape() const { return getImpl()->getShape(); }

mlir::Type RefType::getEleTy() const { return getImpl()->getElementType(); }

void PrintOp::build(Builder *builder, OperationState &result,
                    ArrayRef<Value> args) {
  result.addOperands(args);
}

llvm::BitVector SubscriptRangeAttr::getSubscriptRangeInfo() const {
  return getImpl()->value;
}

SubscriptRangeAttr SubscriptRangeAttr::get(mlir::MLIRContext *ctxt,
                                           llvm::BitVector vector) {
  return Base::get(ctxt, FC_SubscriptRangeAttr, vector);
}

llvm::BitVector StringInfoAttr::getStringInfo() const {
  return getImpl()->value;
}

StringInfoAttr StringInfoAttr::get(mlir::MLIRContext *ctxt,
                                   llvm::BitVector &vector) {
  return Base::get(ctxt, FC_StringInfoAttr, vector);
}

void GlobalOp::build(Builder *builder, OperationState &result, FC::RefType type,
                     bool isConstant, StringRef name, Attribute value) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
  result.types.push_back(type);
  result.addAttribute("type", TypeAttr::get(type));
  if (isConstant)
    result.addAttribute("constant", builder->getUnitAttr());
  if (value)
    result.addAttribute("value", value);
  result.addRegion();
}

void FCLoadOp::build(Builder *builder, OperationState &result, Value pointerVal,
                     ArrayRef<SubscriptRange> indices) {
  result.addOperands(pointerVal);
  auto eleTy = pointerVal.getType().cast<FC::RefType>().getEleTy();
  if (indices.empty()) {
    result.addTypes(eleTy);
    return;
  }
  auto arrTy = eleTy.cast<FC::ArrayType>();

  FC::ArrayType::Shape shape;

  // FIXME: Subscript range is linearly stored as operands
  //. There is no way to retrieve the orignal SubscriptRange
  // passed by the user. Hence, we are using the
  // SubscriptRangeAttr to hold that info and get the actual
  // Range info back. This has been done in StoreOp as well.
  llvm::BitVector infoList;
  for (auto rangeIndex : llvm::enumerate(indices)) {
    auto &range = rangeIndex.value();
    if (!range.isRangeType()) {
      result.addOperands(range.subscript());
      infoList.push_back(true);  // only lower bound.
      infoList.push_back(false); // no upper bound
      infoList.push_back(false); // no stride.
      continue;
    }
    // TODO: update static types.
    shape.push_back(ArrayType::Dim::getUnknown());
    auto lb = range.lowerBound();
    auto ub = range.upperBound();
    auto stride = range.stride();
    if (lb) {
      result.addOperands(lb);
    }
    if (ub) {
      result.addOperands(ub);
    }
    if (stride) {
      result.addOperands(stride);
    }
    infoList.push_back(lb != nullptr);     // only lower bound.
    infoList.push_back(stride != nullptr); // no upper bound
    infoList.push_back(ub != nullptr);     // no stride.
  }
  result.addAttribute("range_info", FC::SubscriptRangeAttr::get(
                                        builder->getContext(), infoList));

  auto resultArrTy = arrTy.getEleTy();
  if (!shape.empty())
    resultArrTy = FC::ArrayType::get(shape, resultArrTy);
  result.addTypes(resultArrTy);
}

void FCStoreOp::build(Builder *builder, OperationState &result,
                      Value valueToStore, Value pointerVal,
                      ArrayRef<SubscriptRange> indices) {
  result.addOperands(valueToStore);
  result.addOperands(pointerVal);
  if (indices.empty()) {
    return;
  }
  llvm::BitVector infoList;
  for (auto rangeIndex : llvm::enumerate(indices)) {
    auto &range = rangeIndex.value();
    if (!range.isRangeType()) {
      result.addOperands(range.subscript());
      infoList.push_back(true);  // only lower bound.
      infoList.push_back(false); // no upper bound
      infoList.push_back(false); // no stride.
      continue;
    }
    // TODO: update static types.
    auto lb = range.lowerBound();
    auto ub = range.upperBound();
    auto stride = range.stride();
    if (lb) {
      result.addOperands(lb);
    }
    if (ub) {
      result.addOperands(ub);
    }
    if (stride) {
      result.addOperands(stride);
    }
    infoList.push_back(lb != nullptr);     // only lower bound.
    infoList.push_back(stride != nullptr); // no upper bound
    infoList.push_back(ub != nullptr);     // no stride.
  }
  result.addAttribute("range_info", FC::SubscriptRangeAttr::get(
                                        builder->getContext(), infoList));
}

void CastToMemRefOp::build(Builder *builder, OperationState &result,
                           Value fcRef) {
  result.addOperands(fcRef);
  auto fcRefType = fcRef.getType().cast<FC::RefType>();
  auto eleTy = fcRefType.getEleTy();

  llvm::SmallVector<int64_t, 2> shape;
  if (auto fcArrType = eleTy.dyn_cast_or_null<FC::ArrayType>()) {
    eleTy = fcArrType.getEleTy();

    auto dims = fcArrType.getShape();
    if (!fcRefType.isStatic()) {
      auto numBounds = dims.size();
      llvm::SmallVector<int64_t, 2> shape(numBounds, -1);
      llvm::SmallVector<int64_t, 2> strides(
          numBounds, mlir::MemRefType::getDynamicStrideOrOffset());
      int64_t offset = mlir::MemRefType::getDynamicStrideOrOffset();
      auto affineMap =
          mlir::makeStridedLinearLayoutMap(strides, offset, fcRef.getContext());
      auto memRefType = mlir::MemRefType::get(shape, eleTy, affineMap);
      result.addTypes(memRefType);
      return;
    }

    for (auto &dim : dims) {
      shape.push_back((dim.upperBound - dim.lowerBound) + 1);
    }

    llvm::SmallVector<mlir::AffineExpr, 2> exprs;
    unsigned dimVal = 0;
    mlir::AffineExpr strideExpr;
    unsigned long sizeTillNow = 1, offsetTillNow = 0;

    for (unsigned I = 0; I < dims.size(); ++I) {
      auto dim = builder->getAffineDimExpr(dimVal++);
      if (I == 0) {
        strideExpr = dim;
        offsetTillNow = -dims[I].lowerBound;
        continue;
      }
      sizeTillNow *= shape[I - 1];
      offsetTillNow += -(dims[I].lowerBound * sizeTillNow);
      auto sizeVal = builder->getAffineConstantExpr(sizeTillNow);
      auto currExpr =
          mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::Mul, dim, sizeVal);
      strideExpr = mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::Add,
                                               currExpr, strideExpr);
    }
    strideExpr = mlir::getAffineBinaryOpExpr(
        mlir::AffineExprKind::Add, strideExpr,
        builder->getAffineConstantExpr(offsetTillNow));

    auto affineMap = mlir::AffineMap::get(shape.size(), 0, {strideExpr});
    auto memRefType = MemRefType::get(shape, eleTy, affineMap);
    result.addTypes(memRefType);
    return;
  }

  auto memRefType = MemRefType::get(shape, eleTy);
  result.addTypes(memRefType);
}

GlobalOp AddressOfOp::getGlobal() {
  auto module = getParentOfType<mlir::ModuleOp>();
  assert(module && "unexpected operation outside of a module");
  return module.lookupSymbol<GlobalOp>(global_name());
}

void FC::FCOpsDialect::printType(mlir::Type ty,
                                 mlir::DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  switch (ty.getKind()) {
  case FC_Ref: {
    auto ptr = ty.cast<FC::RefType>();
    auto eleTy = ptr.getEleTy();
    os << "ref<";
    p.printType(eleTy);
    os << ">";
  } break;
  case FC_Array: {
    auto array = ty.cast<FC::ArrayType>();
    auto eleTy = array.getEleTy();
    os << "array<";
    for (auto &dim : array.getShape()) {
      if (dim.size == -1) {
        os << "? x ";
      } else {
        os << dim.lowerBound << ":" << dim.upperBound << " x ";
      }
    }
    p.printType(eleTy);
    os << ">";
  } break;
  default:
    llvm_unreachable("Unknown FC dialect type");
  }
}

int64_t LBoundOp::getDim() {
  auto dimVal =
      llvm::dyn_cast_or_null<mlir::ConstantIndexOp>(dim().getDefiningOp());
  assert(dimVal);
  return dimVal.getValue();
}

int64_t UBoundOp::getDim() {
  auto dimVal =
      llvm::dyn_cast_or_null<mlir::ConstantIndexOp>(dim().getDefiningOp());
  assert(dimVal);
  return dimVal.getValue();
}

void FCOpsDialect::printAttribute(mlir::Attribute attr,
                                  mlir::DialectAsmPrinter &p) const {
  if (auto symAttr = attr.dyn_cast<SubscriptRangeAttr>()) {
    p << FC::SubscriptRangeAttr::getAttrName() << "< ";
    for (auto bit : symAttr.getSubscriptRangeInfo().set_bits()) {
      p << bit << " ";
    }
    p << ">";
    return;
  }
  if (auto symAttr = attr.dyn_cast<StringInfoAttr>()) {
    p << FC::StringInfoAttr::getAttrName() << "< ";
    for (auto bit : symAttr.getStringInfo().set_bits()) {
      p << bit << " ";
    }
    p << ">";
    return;
  }
  assert(false && "unknown attribute type to print");
}

// NOTE: Code copied from MLIR

// Returns an array of mnemonics for CmpFPredicates indexed by values thereof.
static inline const char *const *getCmpFPredicateNames() {
  static const char *predicateNames[] = {
      /*AlwaysFalse*/ "false",
      /*OEQ*/ "oeq",
      /*OGT*/ "ogt",
      /*OGE*/ "oge",
      /*OLT*/ "olt",
      /*OLE*/ "ole",
      /*ONE*/ "one",
      /*ORD*/ "ord",
      /*UEQ*/ "ueq",
      /*UGT*/ "ugt",
      /*UGE*/ "uge",
      /*ULT*/ "ult",
      /*ULE*/ "ule",
      /*UNE*/ "une",
      /*UNO*/ "uno",
      /*AlwaysTrue*/ "true",
  };
  static_assert(std::extent<decltype(predicateNames)>::value ==
                    (size_t)CmpFPredicate::NumPredicates,
                "wrong number of predicate names");
  return predicateNames;
}
// NOTE: Copy from MLIR source end.

Region &DoOp::getLoopBody() { return region(); }

bool DoOp::isDefinedOutsideOfLoop(Value value) {
  return !region().isAncestor(value.getParentRegion());
}

LogicalResult DoOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (auto *op : ops)
    op->moveBefore(this->getOperation());
  return success();
}

Block *FCFuncOp::addEntryBlock() {
  auto *entry = new Block();
  push_back(entry);
  entry->addArguments(getType().getInputs());
  return entry;
}

void FCFuncOp::build(Builder *builder, OperationState &result, StringRef name,
                     FunctionType type, ArrayRef<NamedAttribute> attrs) {
  result.addRegion();
  result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
  result.addAttribute("type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
}

FCFuncOp FCFuncOp::create(Location location, StringRef name, FunctionType type,
                          ArrayRef<NamedAttribute> attrs) {
  OperationState state(location, "fc.function");
  Builder builder(location->getContext());
  FC::FCFuncOp::build(&builder, state, name, type, attrs);
  return llvm::cast<FCFuncOp>(Operation::create(state));
}

void FCFuncOp::addNestedFunction(FC::FCFuncOp funcOp) {
  auto &entryBlock = front();
  entryBlock.push_front(funcOp);
}

void FCCallOp::build(Builder *builder, OperationState &result, FCFuncOp callee,
                     ArrayRef<Value> operands) {
  auto sym = builder->getSymbolRefAttr(callee);
  FCCallOp::build(builder, result, sym, callee.getType().getResults(),
                  operands);
}

void FCCallOp::build(Builder *builder, OperationState &result,
                     SymbolRefAttr symbolScopeList, ArrayRef<Type> results,
                     ArrayRef<Value> operands) {
  result.addOperands(operands);
  result.addAttribute("symbol_scope", symbolScopeList);
  result.addAttribute("num_operands",
                      builder->getI32IntegerAttr(operands.size()));
  result.addTypes(results);
}

void GetElementRefOp::build(Builder *builder, OperationState &result,
                            SymbolRefAttr symbolScopeList, Type resultType) {
  result.addAttribute("symbol_scope", symbolScopeList);
  result.addTypes(resultType);
}

#define GET_OP_CLASSES
#include "dialect/FCOps/FCOps.cpp.inc"
