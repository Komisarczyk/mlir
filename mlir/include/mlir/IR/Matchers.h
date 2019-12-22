//===- Matchers.h - Various common matchers ---------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a simple and efficient mechanism for performing general
// tree-based pattern matching over MLIR. This mechanism is inspired by LLVM's
// include/llvm/IR/PatternMatch.h.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MATCHERS_H
#define MLIR_MATCHERS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"

namespace mlir {
namespace detail {

/// The matcher that matches a certain kind of Attribute and binds the value
/// inside the Attribute.
template <
    typename AttrClass,
    // Require AttrClass to be a derived class from Attribute and get its
    // value type
    typename ValueType =
        typename std::enable_if<std::is_base_of<Attribute, AttrClass>::value,
                                AttrClass>::type::ValueType,
    // Require the ValueType is not void
    typename = typename std::enable_if<!std::is_void<ValueType>::value>::type>
struct attr_value_binder {
  ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  attr_value_binder(ValueType *bv) : bind_value(bv) {}

  bool match(const Attribute &attr) {
    if (auto intAttr = attr.dyn_cast<AttrClass>()) {
      *bind_value = intAttr.getValue();
      return true;
    }
    return false;
  }
};

/// The matcher that matches a constant foldable operation that has no side
/// effect, no operands and produces a single result.
template <typename AttrT> struct constant_op_binder {
  AttrT *bind_value;

  /// Creates a matcher instance that binds the constant attribute value to
  /// bind_value if match succeeds.
  constant_op_binder(AttrT *bind_value) : bind_value(bind_value) {}

  bool match(Operation *op) {
    if (op->getNumOperands() > 0 || op->getNumResults() != 1)
      return false;
    if (!op->hasNoSideEffect())
      return false;

    SmallVector<OpFoldResult, 1> foldedOp;
    if (succeeded(op->fold(/*operands=*/llvm::None, foldedOp))) {
      if (auto attr = foldedOp.front().dyn_cast<Attribute>()) {
        if ((*bind_value = attr.dyn_cast<AttrT>()))
          return true;
      }
    }
    return false;
  }
};

/// The matcher that matches a constant scalar / vector splat / tensor splat
/// integer operation and binds the constant integer value.
struct constant_int_op_binder {
  IntegerAttr::ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  constant_int_op_binder(IntegerAttr::ValueType *bv) : bind_value(bv) {}

  bool match(Operation *op) {
    Attribute attr;
    if (!constant_op_binder<Attribute>(&attr).match(op))
      return false;
    auto type = op->getResult(0)->getType();

    if (type.isIntOrIndex()) {
      return attr_value_binder<IntegerAttr>(bind_value).match(attr);
    }
    if (type.isa<VectorType>() || type.isa<RankedTensorType>()) {
      if (auto splatAttr = attr.dyn_cast<SplatElementsAttr>()) {
        return attr_value_binder<IntegerAttr>(bind_value)
            .match(splatAttr.getSplatValue());
      }
    }
    return false;
  }
};

/// The matcher that matches a given target constant scalar / vector splat /
/// tensor splat integer value.
template <int64_t TargetValue> struct constant_int_value_matcher {
  bool match(Operation *op) {
    APInt value;
    return constant_int_op_binder(&value).match(op) && TargetValue == value;
  }
};

/// The matcher that matches anything except the given target constant scalar /
/// vector splat / tensor splat integer value.
template <int64_t TargetNotValue> struct constant_int_not_value_matcher {
  bool match(Operation *op) {
    APInt value;
    return constant_int_op_binder(&value).match(op) && TargetNotValue != value;
  }
};

/// The matcher that matches a certain kind of op.
template <typename OpClass> struct op_matcher {
  bool match(Operation *op) { return isa<OpClass>(op); }
};

/// Trait to check whether T provides a 'match' method with type
/// `OperationOrValue`.
template <typename T, typename OperationOrValue>
using has_operation_or_value_matcher_t =
    decltype(std::declval<T>().match(std::declval<OperationOrValue>()));

/// Statically switch to a Value matcher.
template <typename MatcherClass>
typename std::enable_if_t<is_detected<detail::has_operation_or_value_matcher_t,
                                      MatcherClass, Value>::value,
                          bool>
matchOperandOrValueAtIndex(Operation *op, unsigned idx, MatcherClass &matcher) {
  return matcher.match(op->getOperand(idx));
}

/// Statically switch to an Operation matcher.
template <typename MatcherClass>
typename std::enable_if_t<is_detected<detail::has_operation_or_value_matcher_t,
                                      MatcherClass, Operation *>::value,
                          bool>
matchOperandOrValueAtIndex(Operation *op, unsigned idx, MatcherClass &matcher) {
  if (auto defOp = op->getOperand(idx)->getDefiningOp())
    return matcher.match(defOp);
  return false;
}

/// Terminal matcher, always returns true.
struct AnyValueMatcher {
  bool match(Value op) const { return true; }
};

/// Binds to a specific value and matches it.
struct PatternMatcherValue {
  PatternMatcherValue(Value val) : value(val) {}
  bool match(Value val) const { return val == value; }
  Value value;
};

struct PatternMatcherAndBindValue {
  PatternMatcherAndBindValue(Value *&val) : value(val) {}
  bool match(Value *v) {
    if (auto *candidateV = dyn_cast<Value>(v)) {
      value = candidateV;
      return true;
    }
    return false;
  }
  Value *&value;
};

template <typename TupleT, class CallbackT, std::size_t... Is>
constexpr void enumerateImpl(TupleT &&tuple, CallbackT &&callback,
                             std::index_sequence<Is...>) {
  (void)std::initializer_list<int>{
      0,
      (callback(std::integral_constant<std::size_t, Is>{}, std::get<Is>(tuple)),
       0)...};
}

template <typename... Tys, typename CallbackT>
constexpr void enumerate(std::tuple<Tys...> &tuple, CallbackT &&callback) {
  detail::enumerateImpl(tuple, std::forward<CallbackT>(callback),
                        std::make_index_sequence<sizeof...(Tys)>{});
}

/// RecursivePatternMatcher that composes.
template <typename OpType, typename... OperandMatchers>
struct RecursivePatternMatcher {
  RecursivePatternMatcher(OperandMatchers... matchers)
      : operandMatchers(matchers...) {}
  bool match(Operation *op) {
    if (!isa<OpType>(op) || op->getNumOperands() != sizeof...(OperandMatchers))
      return false;
    bool res = true;
    enumerate(operandMatchers, [&](size_t index, auto &matcher) {
      res &= matchOperandOrValueAtIndex(op, index, matcher);
    });
    return res;
  }
  std::tuple<OperandMatchers...> operandMatchers;
};

} // end namespace detail

/// Matches a value from a constant foldable operation and writes the value to
/// bind_value.
template <typename AttrT>
inline detail::constant_op_binder<AttrT> m_Constant(AttrT *bind_value) {
  return detail::constant_op_binder<AttrT>(bind_value);
}

/// Matches a constant scalar / vector splat / tensor splat integer one.
inline detail::constant_int_value_matcher<1> m_One() {
  return detail::constant_int_value_matcher<1>();
}

/// Matches the given OpClass.
template <typename OpClass> inline detail::op_matcher<OpClass> m_Op() {
  return detail::op_matcher<OpClass>();
}

/// Matches a constant scalar / vector splat / tensor splat integer zero.
inline detail::constant_int_value_matcher<0> m_Zero() {
  return detail::constant_int_value_matcher<0>();
}

/// Matches a constant scalar / vector splat / tensor splat integer that is any
/// non-zero value.
inline detail::constant_int_not_value_matcher<0> m_NonZero() {
  return detail::constant_int_not_value_matcher<0>();
}

/// Entry point for matching a pattern over a Value.
template <typename Pattern>
inline bool matchPattern(Value value, const Pattern &pattern) {
  // TODO: handle other cases
  if (auto *op = value->getDefiningOp())
    return const_cast<Pattern &>(pattern).match(op);
  return false;
}

/// Entry point for matching a pattern over an Operation.
template <typename Pattern>
inline bool matchPattern(Operation *op, const Pattern &pattern) {
  return const_cast<Pattern &>(pattern).match(op);
}

/// Matches a constant holding a scalar/vector/tensor integer (splat) and
/// writes the integer value to bind_value.
inline detail::constant_int_op_binder
m_ConstantInt(IntegerAttr::ValueType *bind_value) {
  return detail::constant_int_op_binder(bind_value);
}

template <typename OpType, typename... Matchers>
auto m_Op(Matchers... matchers) {
  return detail::RecursivePatternMatcher<OpType, Matchers...>(matchers...);
}

namespace matchers {
inline auto m_Any() { return detail::AnyValueMatcher(); }
inline auto m_SpecificVal(Value *v) { return detail::PatternMatcherValue(v); }
inline auto m_Val(Value *&v) { return detail::PatternMatcherAndBindValue(v); }

class AffinePattern {
  public:
    MLIRContext *ctx_;
    AffineExpr expr_;
    int64_t constant_;
    int64_t coefficient_;

  public:
    AffinePattern() = delete; 
    AffinePattern(MLIRContext *ctx) : ctx_(ctx), expr_(AffineExpr()),
      constant_(0), coefficient_(1) {};
};

class m_Placeholder {
  public:
    AffinePattern pattern_;
    detail::PatternMatcherValue candidate_;

  public:
    m_Placeholder() = delete;
    m_Placeholder(MLIRContext* ctx, detail::PatternMatcherValue candidate) : 
      pattern_(AffinePattern(ctx)), candidate_(candidate) {};
};

class StructuredArrayPlaceholder {
  public:
    StructuredArrayPlaceholder(detail::PatternMatcherAndBindValue value) : 
      placeholders_({}), bindValue_(value) {};
    StructuredArrayPlaceholder operator()(SmallVector<m_Placeholder, 4> indexings) {
      StructuredArrayPlaceholder placeholder =
        StructuredArrayPlaceholder(this->bindValue_);
      placeholder.placeholders_.clear();
      placeholder.placeholders_ = indexings;
      return placeholder;
    }
    SmallVector<m_Placeholder, 4> placeholders() {
      return placeholders_;
    }

  private:
    SmallVector<m_Placeholder, 4> placeholders_;
    detail::PatternMatcherAndBindValue bindValue_; 
};
using m_ArrayPlaceholder = StructuredArrayPlaceholder; 

inline m_Placeholder
    operator+(m_Placeholder p, int64_t i) {
  p.pattern_.constant_ += i; 
  return p;
}

inline m_Placeholder
    operator-(m_Placeholder p, int64_t i) {
  p.pattern_.constant_ -= i;
  return p;
}

inline m_Placeholder
    operator*(int64_t i, m_Placeholder p) {
  if (i <= 0)
    llvm_unreachable("Invalid coefficient for Placeholder");
  p.pattern_.coefficient_ *= i;
  return p;
}

inline m_Placeholder
    operator*(m_Placeholder p, int64_t i) {
  if (i <= 0)
    llvm_unreachable("Invalid coefficient for Placeholder");
  p.pattern_.coefficient_ *= i;
  return p;
}

} // namespace matchers

namespace detail {

template <typename OpClass> struct op_load_store_matcher {

  SmallVector<matchers::m_Placeholder, 4> placeholders_;

  op_load_store_matcher(SmallVector<matchers::m_Placeholder, 4> ps) : placeholders_(ps) {
    int pos = 0;
    for (auto &placeholder : placeholders_) { 
      // At this point we know the placeholder postion.
      // We create the affine expression to match.
      detail::bindDims(
        placeholder.pattern_.ctx_, 
        placeholder.pattern_.expr_, pos++);
      placeholder.pattern_.expr_ = placeholder.pattern_.expr_ + 
        placeholder.pattern_.constant_;
      placeholder.pattern_.expr_ = placeholder.pattern_.expr_ *
        placeholder.pattern_.coefficient_;
    }
  };
  op_load_store_matcher() = delete;
  bool match(Operation *op) { 
    if (!placeholders_.size()) {
      llvm_unreachable("expect non empty placeholders");
    }
    if (!op) {
      return false;
    }
    if (!isa<OpClass>(op)) {
      return false;
    }
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      size_t dims = loadOp.getAffineMap().getNumResults();
      if (dims != placeholders_.size()) {
        return false;
      }
      SmallVector<Value *, 4> operands = loadOp.getMapOperands();
      for (size_t dim = 0; dim < dims; dim++) {
        AffineExpr loadAffine = loadOp.getAffineMap().getResult(dim);
        if (placeholders_[dim].pattern_.expr_ != loadAffine) {
          return false;
        }
        if (!placeholders_[dim].candidate_.match(operands[dim])) {
          return false;
        }
      }
      return true;
    }
    if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      size_t dims = storeOp.getAffineMap().getNumResults();
      if (dims != placeholders_.size()) {
        return false;
      }
      SmallVector<Value *, 4> operands = storeOp.getMapOperands();
      for (size_t dim = 0; dim < dims; dim++) {
        AffineExpr storeAffine = storeOp.getAffineMap().getResult(dim);
        if (placeholders_[dim].pattern_.expr_ != storeAffine) {
          return false; 
        }
        if (!placeholders_[dim].candidate_.match(operands[dim])) {
          return false;
        }
      }
      return true;
    }
    llvm_unreachable("expect AffineStore or AffineLoad");
  };
};


template <typename OpClass> struct op_load_store_array_matcher {
    matchers::StructuredArrayPlaceholder arrayPlaceholder_;

    op_load_store_array_matcher(matchers::StructuredArrayPlaceholder a) :
      arrayPlaceholder_(a) {};
    op_load_store_array_matcher() = delete;

    bool match(Operation *op) { 
      auto placeholderMatcher = 
        detail::op_load_store_matcher<OpClass>(arrayPlaceholder_.placeholders());
      if (!placeholderMatcher.match(op))
        return false;
      // TODO bind the array name to bindValue_;
      return true; 
    };
};

} // end namespace detail

namespace matchers {

template <typename OpClass>
    inline detail::op_load_store_array_matcher<OpClass> m_Op(StructuredArrayPlaceholder arg) {
  return detail::op_load_store_array_matcher<OpClass>(arg);
}

template<class T, class...>
struct are_same : std::true_type
{};

template<class T, class U, class... TT>
struct are_same<T, U, TT...>
    : std::integral_constant<bool, std::is_same<T,U>{} && are_same<T, TT...>{}>
{};

template <typename OpClass, typename... Args> 
    inline detail::op_load_store_matcher<OpClass> m_Op(m_Placeholder arg, Args... args) {
  static_assert(are_same<m_Placeholder, Args...>{},
    "all args must be Placeholder");
  return detail::op_load_store_matcher<OpClass>({arg, args...});
}
 
} // end namespace matchers.

} // end namespace mlir

#endif // MLIR_MATCHERS_H
