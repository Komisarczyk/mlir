/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

class AddOp;
class CastOp;
class ConstantOp;
class GenericCallOp;
class MulOp;
class PrintOp;
class ReshapeOp;
class ReturnOp;
class TransposeOp;

//===----------------------------------------------------------------------===//
// toy::AddOp declarations
//===----------------------------------------------------------------------===//

class AddOpOperandAdaptor {
public:
  AddOpOperandAdaptor(ArrayRef<Value> values);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value lhs();
  Value rhs();

private:
  ArrayRef<Value> tblgen_operands;
};
class AddOp : public Op<AddOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::NOperands<2>::Impl, MemoryEffectOpInterface::Trait, ShapeInference::Trait> {
public:
  using Op::Op;
  using OperandAdaptor = AddOpOperandAdaptor;
  static StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  Operation::operand_range getODSOperands(unsigned index);
  Value lhs();
  Value rhs();
  ::mlir::MutableOperandRange lhsMutable();
  ::mlir::MutableOperandRange rhsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  static void build(OpBuilder &b, OperationState &state, Value lhs, Value rhs);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs);
  static void build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  void inferShapes();
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects);
};

//===----------------------------------------------------------------------===//
// toy::CastOp declarations
//===----------------------------------------------------------------------===//

class CastOpOperandAdaptor {
public:
  CastOpOperandAdaptor(ArrayRef<Value> values);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value input();

private:
  ArrayRef<Value> tblgen_operands;
};
class CastOp : public Op<CastOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::OneOperand, ShapeInference::Trait, MemoryEffectOpInterface::Trait, OpTrait::SameOperandsAndResultShape> {
public:
  using Op::Op;
  using OperandAdaptor = CastOpOperandAdaptor;
  static StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  Operation::operand_range getODSOperands(unsigned index);
  Value input();
  ::mlir::MutableOperandRange inputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  Value output();
  static void build(OpBuilder &odsBuilder, OperationState &odsState, Type output, Value input);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value input);
  static void build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
  void inferShapes();
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects);
};

//===----------------------------------------------------------------------===//
// toy::ConstantOp declarations
//===----------------------------------------------------------------------===//

class ConstantOpOperandAdaptor {
public:
  ConstantOpOperandAdaptor(ArrayRef<Value> values);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ArrayRef<Value> getODSOperands(unsigned index);

private:
  ArrayRef<Value> tblgen_operands;
};
class ConstantOp : public Op<ConstantOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::ZeroOperands, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using OperandAdaptor = ConstantOpOperandAdaptor;
  static StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  DenseElementsAttr valueAttr();
  DenseElementsAttr value();
  void valueAttr(DenseElementsAttr attr);
  static void build(OpBuilder &builder, OperationState &state, DenseElementsAttr value);
  static void build(OpBuilder &builder, OperationState &state, double value);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, DenseElementsAttr value);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, DenseElementsAttr value);
  static void build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects);
};

//===----------------------------------------------------------------------===//
// toy::GenericCallOp declarations
//===----------------------------------------------------------------------===//

class GenericCallOpOperandAdaptor {
public:
  GenericCallOpOperandAdaptor(ArrayRef<Value> values);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ArrayRef<Value> getODSOperands(unsigned index);
  ArrayRef<Value> inputs();

private:
  ArrayRef<Value> tblgen_operands;
};
class GenericCallOp : public Op<GenericCallOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::VariadicOperands, CallOpInterface::Trait> {
public:
  using Op::Op;
  using OperandAdaptor = GenericCallOpOperandAdaptor;
  static StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  Operation::operand_range getODSOperands(unsigned index);
  Operation::operand_range inputs();
  ::mlir::MutableOperandRange inputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  FlatSymbolRefAttr calleeAttr();
  StringRef callee();
  void calleeAttr(FlatSymbolRefAttr attr);
  static void build(OpBuilder &builder, OperationState &state, StringRef callee, ArrayRef<Value> arguments);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, FlatSymbolRefAttr callee, ValueRange inputs);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, FlatSymbolRefAttr callee, ValueRange inputs);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, StringRef callee, ValueRange inputs);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, StringRef callee, ValueRange inputs);
  static void build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
  CallInterfaceCallable getCallableForCallee();
  Operation::operand_range getArgOperands();
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
};

//===----------------------------------------------------------------------===//
// toy::MulOp declarations
//===----------------------------------------------------------------------===//

class MulOpOperandAdaptor {
public:
  MulOpOperandAdaptor(ArrayRef<Value> values);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value lhs();
  Value rhs();

private:
  ArrayRef<Value> tblgen_operands;
};
class MulOp : public Op<MulOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::NOperands<2>::Impl, MemoryEffectOpInterface::Trait, ShapeInference::Trait> {
public:
  using Op::Op;
  using OperandAdaptor = MulOpOperandAdaptor;
  static StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  Operation::operand_range getODSOperands(unsigned index);
  Value lhs();
  Value rhs();
  ::mlir::MutableOperandRange lhsMutable();
  ::mlir::MutableOperandRange rhsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  static void build(OpBuilder &b, OperationState &state, Value lhs, Value rhs);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs);
  static void build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  void inferShapes();
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects);
};

//===----------------------------------------------------------------------===//
// toy::PrintOp declarations
//===----------------------------------------------------------------------===//

class PrintOpOperandAdaptor {
public:
  PrintOpOperandAdaptor(ArrayRef<Value> values);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value input();

private:
  ArrayRef<Value> tblgen_operands;
};
class PrintOp : public Op<PrintOp, OpTrait::ZeroRegion, OpTrait::ZeroResult, OpTrait::ZeroSuccessor, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = PrintOpOperandAdaptor;
  static StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  Operation::operand_range getODSOperands(unsigned index);
  Value input();
  ::mlir::MutableOperandRange inputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, Value input);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value input);
  static void build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
};

//===----------------------------------------------------------------------===//
// toy::ReshapeOp declarations
//===----------------------------------------------------------------------===//

class ReshapeOpOperandAdaptor {
public:
  ReshapeOpOperandAdaptor(ArrayRef<Value> values);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value input();

private:
  ArrayRef<Value> tblgen_operands;
};
class ReshapeOp : public Op<ReshapeOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::OneOperand, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using OperandAdaptor = ReshapeOpOperandAdaptor;
  static StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  Operation::operand_range getODSOperands(unsigned index);
  Value input();
  ::mlir::MutableOperandRange inputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Value input);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value input);
  static void build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
  static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects);
};

//===----------------------------------------------------------------------===//
// toy::ReturnOp declarations
//===----------------------------------------------------------------------===//

class ReturnOpOperandAdaptor {
public:
  ReturnOpOperandAdaptor(ArrayRef<Value> values);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ArrayRef<Value> getODSOperands(unsigned index);
  ArrayRef<Value> input();

private:
  ArrayRef<Value> tblgen_operands;
};
class ReturnOp : public Op<ReturnOp, OpTrait::ZeroRegion, OpTrait::ZeroResult, OpTrait::ZeroSuccessor, OpTrait::VariadicOperands, MemoryEffectOpInterface::Trait, OpTrait::HasParent<FuncOp>::Impl, OpTrait::IsTerminator> {
public:
  using Op::Op;
  using OperandAdaptor = ReturnOpOperandAdaptor;
  static StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  Operation::operand_range getODSOperands(unsigned index);
  Operation::operand_range input();
  ::mlir::MutableOperandRange inputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  static void build(OpBuilder &b, OperationState &state);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, ValueRange input);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange input);
  static void build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects);

    bool hasOperand() { return getNumOperands() != 0; }
  
};

//===----------------------------------------------------------------------===//
// toy::TransposeOp declarations
//===----------------------------------------------------------------------===//

class TransposeOpOperandAdaptor {
public:
  TransposeOpOperandAdaptor(ArrayRef<Value> values);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value input();

private:
  ArrayRef<Value> tblgen_operands;
};
class TransposeOp : public Op<TransposeOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::OneOperand, MemoryEffectOpInterface::Trait, ShapeInference::Trait> {
public:
  using Op::Op;
  using OperandAdaptor = TransposeOpOperandAdaptor;
  static StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  Operation::operand_range getODSOperands(unsigned index);
  Value input();
  ::mlir::MutableOperandRange inputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  static void build(OpBuilder &b, OperationState &state, Value input);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Value input);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value input);
  static void build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
  static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);
  void inferShapes();
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects);
};

#endif  // GET_OP_CLASSES

