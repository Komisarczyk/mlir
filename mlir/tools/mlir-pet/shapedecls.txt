/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Operation Interface Declarations                                           *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace detail {
struct ShapeInferenceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual void inferShapes(Operation *tablegen_opaque_op) = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
public:
    void inferShapes(Operation *tablegen_opaque_op) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;
      return op.inferShapes();
    }
  };
};
} // end namespace detail
class ShapeInference : public OpInterface<ShapeInference, detail::ShapeInferenceInterfaceTraits> {
public:
  using OpInterface<ShapeInference, detail::ShapeInferenceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct ShapeInferenceTrait : public OpInterface<ShapeInference, detail::ShapeInferenceInterfaceTraits>::Trait<ConcreteOp> {
  };
    template <typename ConcreteOp>
    struct Trait : public ShapeInferenceTrait<ConcreteOp> {};
  void inferShapes();
};
