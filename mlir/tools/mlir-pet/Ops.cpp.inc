/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

toy::AddOp,
toy::CastOp,
toy::ConstantOp,
toy::GenericCallOp,
toy::MulOp,
toy::PrintOp,
toy::ReshapeOp,
toy::ReturnOp,
toy::TransposeOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// toy::AddOp definitions
//===----------------------------------------------------------------------===//

AddOpOperandAdaptor::AddOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

std::pair<unsigned, unsigned> AddOpOperandAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

ArrayRef<Value> AddOpOperandAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(tblgen_operands.begin(), valueRange.first),
           std::next(tblgen_operands.begin(), valueRange.first + valueRange.second)};
}

Value AddOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value AddOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef AddOp::getOperationName() {
  return "toy.add";
}

std::pair<unsigned, unsigned> AddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::operand_range AddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

Value AddOp::lhs() {
  return *getODSOperands(0).begin();
}

Value AddOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AddOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AddOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::result_range AddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}



void AddOp::build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void AddOp::build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddOp::build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult AddOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseBinaryOp(parser, result);
}

void AddOp::print(OpAsmPrinter &p) {
  return ::printBinaryOp(p, *this);
}

LogicalResult AddOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64())))) {
        return emitOpError("result #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void AddOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// toy::CastOp definitions
//===----------------------------------------------------------------------===//

CastOpOperandAdaptor::CastOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

std::pair<unsigned, unsigned> CastOpOperandAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

ArrayRef<Value> CastOpOperandAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(tblgen_operands.begin(), valueRange.first),
           std::next(tblgen_operands.begin(), valueRange.first + valueRange.second)};
}

Value CastOpOperandAdaptor::input() {
  return *getODSOperands(0).begin();
}

StringRef CastOp::getOperationName() {
  return "toy.cast";
}

std::pair<unsigned, unsigned> CastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::operand_range CastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

Value CastOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CastOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::result_range CastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

Value CastOp::output() {
  return *getODSResults(0).begin();
}

void CastOp::build(OpBuilder &odsBuilder, OperationState &odsState, Type output, Value input) {
  odsState.addOperands(input);
  odsState.addTypes(output);
}

void CastOp::build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CastOp::build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult CastOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64())))) {
        return emitOpError("result #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}





ParseResult CastOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType inputRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> inputOperands(inputRawOperands);  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  (void)inputOperandsLoc;
  Type inputRawTypes[1];
  ArrayRef<Type> inputTypes(inputRawTypes);
  Type outputRawTypes[1];
  ArrayRef<Type> outputTypes(outputRawTypes);

  if (parser.parseOperand(inputRawOperands[0]))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(inputRawTypes[0]))
    return failure();
  if (parser.parseKeyword("to"))
    return failure();

  if (parser.parseType(outputRawTypes[0]))
    return failure();
  result.addTypes(outputTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return failure();
  return success();
}

void CastOp::print(OpAsmPrinter &p) {
  p << "toy.cast";
  p << " ";
  p << input();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(input().getType());
  p << " " << "to";
  p << " ";
  p << ArrayRef<Type>(output().getType());
}

void CastOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// toy::ConstantOp definitions
//===----------------------------------------------------------------------===//

ConstantOpOperandAdaptor::ConstantOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

std::pair<unsigned, unsigned> ConstantOpOperandAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

ArrayRef<Value> ConstantOpOperandAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(tblgen_operands.begin(), valueRange.first),
           std::next(tblgen_operands.begin(), valueRange.first + valueRange.second)};
}

StringRef ConstantOp::getOperationName() {
  return "toy.constant";
}

std::pair<unsigned, unsigned> ConstantOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::operand_range ConstantOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ConstantOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::result_range ConstantOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

DenseElementsAttr ConstantOp::valueAttr() {
  return this->getAttr("value").cast<DenseElementsAttr>();
}

DenseElementsAttr ConstantOp::value() {
  auto attr = valueAttr();
  return attr;
}

void ConstantOp::valueAttr(DenseElementsAttr attr) {
  this->getOperation()->setAttr("value", attr);
}

void ConstantOp::build(OpBuilder &builder, OperationState &state, DenseElementsAttr value) {
      build(builder, state, value.getType(), value);
    
}



void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, DenseElementsAttr value) {
  odsState.addAttribute("value", value);
  odsState.addTypes(resultType0);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, DenseElementsAttr value) {
  odsState.addAttribute("value", value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantOp::build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseConstantOp(parser, result);
}

void ConstantOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult ConstantOp::verify() {
  auto tblgen_value = this->getAttr("value");
  if (!tblgen_value) return emitOpError("requires attribute 'value'");
  {
    if (!((tblgen_value.isa<DenseFPElementsAttr>() &&tblgen_value.cast<DenseElementsAttr>().getType().getElementType().isF64()))) return emitOpError("attribute 'value' failed to satisfy constraint: 64-bit float elements attribute");
  }
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64())))) {
        return emitOpError("result #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}

void ConstantOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// toy::GenericCallOp definitions
//===----------------------------------------------------------------------===//

GenericCallOpOperandAdaptor::GenericCallOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

std::pair<unsigned, unsigned> GenericCallOpOperandAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

ArrayRef<Value> GenericCallOpOperandAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(tblgen_operands.begin(), valueRange.first),
           std::next(tblgen_operands.begin(), valueRange.first + valueRange.second)};
}

ArrayRef<Value> GenericCallOpOperandAdaptor::inputs() {
  return getODSOperands(0);
}

StringRef GenericCallOp::getOperationName() {
  return "toy.generic_call";
}

std::pair<unsigned, unsigned> GenericCallOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

Operation::operand_range GenericCallOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

Operation::operand_range GenericCallOp::inputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange GenericCallOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GenericCallOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::result_range GenericCallOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

FlatSymbolRefAttr GenericCallOp::calleeAttr() {
  return this->getAttr("callee").cast<FlatSymbolRefAttr>();
}

StringRef GenericCallOp::callee() {
  auto attr = calleeAttr();
  return attr.getValue();
}

void GenericCallOp::calleeAttr(FlatSymbolRefAttr attr) {
  this->getOperation()->setAttr("callee", attr);
}



void GenericCallOp::build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, FlatSymbolRefAttr callee, ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute("callee", callee);
  odsState.addTypes(resultType0);
}

void GenericCallOp::build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, FlatSymbolRefAttr callee, ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute("callee", callee);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GenericCallOp::build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, StringRef callee, ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute("callee", odsBuilder.getSymbolRefAttr(callee));
  odsState.addTypes(resultType0);
}

void GenericCallOp::build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, StringRef callee, ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute("callee", odsBuilder.getSymbolRefAttr(callee));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GenericCallOp::build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult GenericCallOp::verify() {
  auto tblgen_callee = this->getAttr("callee");
  if (!tblgen_callee) return emitOpError("requires attribute 'callee'");
  {
    if (!((tblgen_callee.isa<FlatSymbolRefAttr>()))) return emitOpError("attribute 'callee' failed to satisfy constraint: flat symbol reference attribute");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64())))) {
        return emitOpError("result #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}





ParseResult GenericCallOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> inputsOperands;
  llvm::SMLoc inputsOperandsLoc = parser.getCurrentLocation();
  (void)inputsOperandsLoc;
  ArrayRef<Type> inputsTypes;
  ArrayRef<Type> allResultTypes;

  FlatSymbolRefAttr calleeAttr;
  if (parser.parseAttribute(calleeAttr, parser.getBuilder().getType<NoneType>(), "callee", result.attributes))
    return failure();
  if (parser.parseLParen())
    return failure();

  if (parser.parseOperandList(inputsOperands))
    return failure();
  if (parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  FunctionType inputs__allResult_functionType;
  if (parser.parseType(inputs__allResult_functionType))
    return failure();
  inputsTypes = inputs__allResult_functionType.getInputs();
  allResultTypes = inputs__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc, result.operands))
    return failure();
  return success();
}

void GenericCallOp::print(OpAsmPrinter &p) {
  p << "toy.generic_call";
  p << " ";
  p.printAttributeWithoutType(calleeAttr());
  p << "(";
  p << inputs();
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"callee"});
  p << " " << ":";
  p << " ";
  p.printFunctionalType(inputs().getTypes(), getOperation()->getResultTypes());
}


//===----------------------------------------------------------------------===//
// toy::MulOp definitions
//===----------------------------------------------------------------------===//

MulOpOperandAdaptor::MulOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

std::pair<unsigned, unsigned> MulOpOperandAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

ArrayRef<Value> MulOpOperandAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(tblgen_operands.begin(), valueRange.first),
           std::next(tblgen_operands.begin(), valueRange.first + valueRange.second)};
}

Value MulOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value MulOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef MulOp::getOperationName() {
  return "toy.mul";
}

std::pair<unsigned, unsigned> MulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::operand_range MulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

Value MulOp::lhs() {
  return *getODSOperands(0).begin();
}

Value MulOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MulOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MulOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::result_range MulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}



void MulOp::build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void MulOp::build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulOp::build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult MulOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseBinaryOp(parser, result);
}

void MulOp::print(OpAsmPrinter &p) {
  return ::printBinaryOp(p, *this);
}

LogicalResult MulOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64())))) {
        return emitOpError("result #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void MulOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// toy::PrintOp definitions
//===----------------------------------------------------------------------===//

PrintOpOperandAdaptor::PrintOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

std::pair<unsigned, unsigned> PrintOpOperandAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

ArrayRef<Value> PrintOpOperandAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(tblgen_operands.begin(), valueRange.first),
           std::next(tblgen_operands.begin(), valueRange.first + valueRange.second)};
}

Value PrintOpOperandAdaptor::input() {
  return *getODSOperands(0).begin();
}

StringRef PrintOp::getOperationName() {
  return "toy.print";
}

std::pair<unsigned, unsigned> PrintOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::operand_range PrintOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

Value PrintOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange PrintOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> PrintOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::result_range PrintOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void PrintOp::build(OpBuilder &odsBuilder, OperationState &odsState, Value input) {
  odsState.addOperands(input);
}

void PrintOp::build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PrintOp::build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult PrintOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64()))) || (((v.getType().isa<MemRefType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64()))))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values or memref of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return mlir::success();
}

ParseResult PrintOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType inputRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> inputOperands(inputRawOperands);  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  (void)inputOperandsLoc;
  Type inputRawTypes[1];
  ArrayRef<Type> inputTypes(inputRawTypes);

  if (parser.parseOperand(inputRawOperands[0]))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(inputRawTypes[0]))
    return failure();
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return failure();
  return success();
}

void PrintOp::print(OpAsmPrinter &p) {
  p << "toy.print";
  p << " ";
  p << input();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(input().getType());
}


//===----------------------------------------------------------------------===//
// toy::ReshapeOp definitions
//===----------------------------------------------------------------------===//

ReshapeOpOperandAdaptor::ReshapeOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

std::pair<unsigned, unsigned> ReshapeOpOperandAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

ArrayRef<Value> ReshapeOpOperandAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(tblgen_operands.begin(), valueRange.first),
           std::next(tblgen_operands.begin(), valueRange.first + valueRange.second)};
}

Value ReshapeOpOperandAdaptor::input() {
  return *getODSOperands(0).begin();
}

StringRef ReshapeOp::getOperationName() {
  return "toy.reshape";
}

std::pair<unsigned, unsigned> ReshapeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::operand_range ReshapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

Value ReshapeOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReshapeOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReshapeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::result_range ReshapeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReshapeOp::build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Value input) {
  odsState.addOperands(input);
  odsState.addTypes(resultType0);
}

void ReshapeOp::build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReshapeOp::build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult ReshapeOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64()))) && ((v.getType().cast<ShapedType>().hasStaticShape())))) {
        return emitOpError("result #") << index << " must be statically shaped tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



ParseResult ReshapeOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType inputRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> inputOperands(inputRawOperands);  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  (void)inputOperandsLoc;
  Type inputRawTypes[1];
  ArrayRef<Type> inputTypes(inputRawTypes);
  SmallVector<Type, 1> allResultTypes;
  if (parser.parseLParen())
    return failure();

  if (parser.parseOperand(inputRawOperands[0]))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(inputRawTypes[0]))
    return failure();
  if (parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseKeyword("to"))
    return failure();

  if (parser.parseTypeList(allResultTypes))
    return failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return failure();
  return success();
}

void ReshapeOp::print(OpAsmPrinter &p) {
  p << "toy.reshape";
  p << "(";
  p << input();
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(input().getType());
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << "to";
  p << " ";
  p << getOperation()->getResultTypes();
}

void ReshapeOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// toy::ReturnOp definitions
//===----------------------------------------------------------------------===//

ReturnOpOperandAdaptor::ReturnOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

std::pair<unsigned, unsigned> ReturnOpOperandAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

ArrayRef<Value> ReturnOpOperandAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(tblgen_operands.begin(), valueRange.first),
           std::next(tblgen_operands.begin(), valueRange.first + valueRange.second)};
}

ArrayRef<Value> ReturnOpOperandAdaptor::input() {
  return getODSOperands(0);
}

StringRef ReturnOp::getOperationName() {
  return "toy.return";
}

std::pair<unsigned, unsigned> ReturnOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

Operation::operand_range ReturnOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

Operation::operand_range ReturnOp::input() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ReturnOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReturnOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::result_range ReturnOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReturnOp::build(OpBuilder &b, OperationState &state) {
 build(b, state, llvm::None); 
}

void ReturnOp::build(OpBuilder &odsBuilder, OperationState &odsState, ValueRange input) {
  odsState.addOperands(input);
}

void ReturnOp::build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReturnOp::build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult ReturnOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

ParseResult ReturnOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> inputOperands;
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  (void)inputOperandsLoc;
  SmallVector<Type, 1> inputTypes;

  if (parser.parseOperandList(inputOperands))
    return failure();
  if (!inputOperands.empty()) {
  if (parser.parseColon())
    return failure();

  if (parser.parseTypeList(inputTypes))
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return failure();
  return success();
}

void ReturnOp::print(OpAsmPrinter &p) {
  p << "toy.return";
  if (!input().empty()) {
  p << " ";
  p << input();
  p << " " << ":";
  p << " ";
  p << input().getTypes();
  }
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
}

void ReturnOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// toy::TransposeOp definitions
//===----------------------------------------------------------------------===//

TransposeOpOperandAdaptor::TransposeOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

std::pair<unsigned, unsigned> TransposeOpOperandAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

ArrayRef<Value> TransposeOpOperandAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(tblgen_operands.begin(), valueRange.first),
           std::next(tblgen_operands.begin(), valueRange.first + valueRange.second)};
}

Value TransposeOpOperandAdaptor::input() {
  return *getODSOperands(0).begin();
}

StringRef TransposeOp::getOperationName() {
  return "toy.transpose";
}

std::pair<unsigned, unsigned> TransposeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::operand_range TransposeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

Value TransposeOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TransposeOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TransposeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::result_range TransposeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}



void TransposeOp::build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Value input) {
  odsState.addOperands(input);
  odsState.addTypes(resultType0);
}

void TransposeOp::build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeOp::build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult TransposeOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64())))) {
        return emitOpError("operand #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isF64())))) {
        return emitOpError("result #") << index << " must be tensor of 64-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}





ParseResult TransposeOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType inputRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> inputOperands(inputRawOperands);  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  (void)inputOperandsLoc;
  Type inputRawTypes[1];
  ArrayRef<Type> inputTypes(inputRawTypes);
  SmallVector<Type, 1> allResultTypes;
  if (parser.parseLParen())
    return failure();

  if (parser.parseOperand(inputRawOperands[0]))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(inputRawTypes[0]))
    return failure();
  if (parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseKeyword("to"))
    return failure();

  if (parser.parseTypeList(allResultTypes))
    return failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return failure();
  return success();
}

void TransposeOp::print(OpAsmPrinter &p) {
  p << "toy.transpose";
  p << "(";
  p << input();
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(input().getType());
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << "to";
  p << " ";
  p << getOperation()->getResultTypes();
}

void TransposeOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


#endif  // GET_OP_CLASSES

