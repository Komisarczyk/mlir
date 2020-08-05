/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* Generated from:
	/home/konrad/Documents/llvm-project/mlir/tools/mlir-pet/ToyCombine.td:45
*/
struct FoldConstantReshapeOptPattern : public RewritePattern {
  FoldConstantReshapeOptPattern(MLIRContext *context)
      : RewritePattern("toy.reshape", {"toy.constant"}, 2, context) {}

  LogicalResult matchAndRewrite(Operation *op0,
                                     PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used for creating ops
    toy::ReshapeOp res;
    DenseElementsAttr arg;
    Operation *tblgen_ops[2];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = dyn_cast_or_null<toy::ReshapeOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = dyn_cast_or_null<toy::ConstantOp>(op1); (void)castedOp1;
      if (!castedOp1) return failure();
      {
        auto tblgen_attr = op1->getAttrOfType<DenseElementsAttr>("value");(void)tblgen_attr;

    if (!(tblgen_attr)) {
      return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
        diag << "expected op 'toy.constant' to have attribute 'value' of type 'DenseElementsAttr'";
      });
    }
        arg = tblgen_attr;
      }
      tblgen_ops[1] = op1;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    SmallVector<Value, 4> tblgen_repl_values;
    toy::ConstantOp tblgen_ConstantOp_0;
    {
      SmallVector<Value, 4> tblgen_values; (void)tblgen_values;
      SmallVector<NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (arg.reshape(((*res.getODSResults(0).begin()).getType()).cast<ShapedType>())) {  tblgen_attrs.emplace_back(rewriter.getIdentifier("value"), arg.reshape(((*res.getODSResults(0).begin()).getType()).cast<ShapedType>())); }
      SmallVector<Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {tblgen_types.push_back(v.getType()); }
      tblgen_ConstantOp_0 = rewriter.create<toy::ConstantOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }
    
    for (auto v : SmallVector<Value, 4>{ tblgen_ConstantOp_0.getODSResults(0) }) { tblgen_repl_values.push_back(v); }    
    rewriter.replaceOp(op0, tblgen_repl_values);
    return success();
  };
};
/* Generated from:
	/home/konrad/Documents/llvm-project/mlir/tools/mlir-pet/ToyCombine.td:58
*/
struct RedundantReshapeOptPattern : public RewritePattern {
  RedundantReshapeOptPattern(MLIRContext *context)
      : RewritePattern("toy.reshape", {}, 1, context) {}

  LogicalResult matchAndRewrite(Operation *op0,
                                     PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used for creating ops
    toy::ReshapeOp res;
    Operation::operand_range arg(op0->getOperands());
    Operation *tblgen_ops[1];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = dyn_cast_or_null<toy::ReshapeOp>(op0); (void)castedOp0;
    res = castedOp0;
    arg = castedOp0.getODSOperands(0);

    if (!(((*res.getODSResults(0).begin()).getType() == (*arg.begin()).getType()))) {
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'res, arg' failed to satisfy constraint: TypesAreIdentical";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    SmallVector<Value, 4> tblgen_repl_values;
    
    for (auto v : SmallVector<Value, 4>{ arg }) { tblgen_repl_values.push_back(v); }    
    rewriter.replaceOp(op0, tblgen_repl_values);
    return success();
  };
};
/* Generated from:
	/home/konrad/Documents/llvm-project/mlir/tools/mlir-pet/ToyCombine.td:32
*/
struct ReshapeReshapeOptPattern : public RewritePattern {
  ReshapeReshapeOptPattern(MLIRContext *context)
      : RewritePattern("toy.reshape", {"toy.reshape"}, 2, context) {}

  LogicalResult matchAndRewrite(Operation *op0,
                                     PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used for creating ops
    Operation::operand_range arg(op0->getOperands());
    Operation *tblgen_ops[2];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = dyn_cast_or_null<toy::ReshapeOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = dyn_cast_or_null<toy::ReshapeOp>(op1); (void)castedOp1;
      if (!castedOp1) return failure();
      arg = castedOp1.getODSOperands(0);
      tblgen_ops[1] = op1;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    SmallVector<Value, 4> tblgen_repl_values;
    toy::ReshapeOp tblgen_ReshapeOp_0;
    {
      SmallVector<Value, 4> tblgen_values; (void)tblgen_values;
      SmallVector<NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg.begin()));
      SmallVector<Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v : castedOp0.getODSResults(0)) {tblgen_types.push_back(v.getType()); }
      tblgen_ReshapeOp_0 = rewriter.create<toy::ReshapeOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }
    
    for (auto v : SmallVector<Value, 4>{ tblgen_ReshapeOp_0.getODSResults(0) }) { tblgen_repl_values.push_back(v); }    
    rewriter.replaceOp(op0, tblgen_repl_values);
    return success();
  };
};
void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(MLIRContext *context, OwningRewritePatternList *patterns) {
  patterns->insert<FoldConstantReshapeOptPattern>(context);
  patterns->insert<RedundantReshapeOptPattern>(context);
  patterns->insert<ReshapeReshapeOptPattern>(context);
}
