#include "islNodeBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace codegen;
using namespace mlir;
using namespace llvm;
using namespace pet;

// TODO: check loop direction.
static isl::ast_expr getUpperBound(isl::ast_node nodeFor, bool &leqBound) {
  if (isl_ast_node_get_type(nodeFor.get()) != isl_ast_node_for)
    llvm_unreachable("expect a for node");
  isl::ast_expr condition = nodeFor.for_get_cond();
  if (isl_ast_expr_get_type(condition.get()) != isl_ast_expr_op)
    llvm_unreachable("conditional expression is not an atomic upper bound");
  // set the flag for leq to increment symbolic bound
  if (isl_ast_expr_get_op_type(condition.get()) == isl_ast_expr_op_le)
    leqBound = true;
  return condition.get_op_arg(1);
}

static bool isInt(isl::ast_expr expression) {
  return isl_ast_expr_get_type(expression.get()) == isl_ast_expr_int;
}

static int getIntFromIslExpr(isl::ast_expr expression) {
  if (isl_ast_expr_get_type(expression.get()) != isl_ast_expr_int)
    llvm_unreachable("expect isl_ast_expr_int expression");
  auto val = expression.get_val();
  return std::stoi(val.to_str());
}

// Simplistic function that looks for an expression of type coeff * i + inc or i
// + inc.
static AffineExpr getAffineFromIslExpr(isl::ast_expr expr, MLIRContext *ctx) {
  assert(((isl_ast_expr_get_type(expr.get()) == isl_ast_expr_id) ||
          (isl_ast_expr_get_type(expr.get()) == isl_ast_expr_op)) &&
         "expect isl_ast_expr_id or isl_ast_epxr op");
  AffineExpr i;
  bindDims(ctx, i);
  if (isl_ast_expr_get_type(expr.get()) == isl_ast_expr_id)
    return i;

  int coeff = 1;
  int inc = 1;
  auto sumOrMinusExpr = isl_ast_expr_get_op_type(expr.get());
  assert(((sumOrMinusExpr == isl_ast_op_add) ||
          (sumOrMinusExpr == isl_ast_op_minus)) &&
         "expect isl_ast_sum or isl_ast_minus");

  // assume the rhs of the sum is an expr_int.
  auto rhsSum = expr.get_op_arg(1);
  assert((isl_ast_expr_get_type(rhsSum.get()) == isl_ast_expr_int) &&
         "expect an isl_ast_expr_int");
  auto incVal = rhsSum.get_val();
  inc = std::stoi(incVal.to_str());

  // check if we have a nested mul.
  auto mulOrId = expr.get_op_arg(0);
  assert(((isl_ast_expr_get_type(mulOrId.get()) == isl_ast_expr_id) ||
          (isl_ast_expr_get_type(mulOrId.get()) == isl_ast_expr_op)) &&
         "expect isl_ast_expr_id or isl_ast_expr_op");
  if (isl_ast_expr_get_type(mulOrId.get()) == isl_ast_expr_id)
    return i + inc;

  // if so get the value of the mul.
  auto mulType = isl_ast_expr_get_op_type(mulOrId.get());
  assert((mulType == isl_ast_op_mul) && "expect isl_ast_mul");
  auto lhsMul = mulOrId.get_op_arg(0);
  assert((isl_ast_expr_get_type(lhsMul.get()) == isl_ast_expr_int) &&
         "expect an isl_ast_expr_int");
  auto coeffVal = lhsMul.get_val();
  coeff = std::stoi(coeffVal.to_str());
  return coeff * i + inc;
}

// walk an isl::ast_expr looking for an isl_ast_expr_id if
// any.
static void getBoundId(isl::ast_expr expr, std::string &id) {
  if (isl_ast_expr_get_type(expr.get()) == isl_ast_expr_id)
    id = expr.get_id().to_str();
  if (isl_ast_expr_get_type(expr.get()) == isl_ast_expr_int)
    return;
  if (isl_ast_expr_get_type(expr.get()) == isl_ast_expr_op)
    for (int i = 0; i < expr.get_op_n_arg(); i++)
      getBoundId(expr.get_op_arg(i), id);
}

// TODO: See how we can get location information.
// TODO: handle degenerate loop (see isl_ast_node_for_is_degenerate)
// TODO: See how to handle more complex expression in the loop.
void IslNodeBuilder::createFor(isl::ast_node forNode) {
  auto lowerBound = forNode.for_get_init();
  auto increment = forNode.for_get_inc();
  auto iterator = forNode.for_get_iterator();
  auto iteratorId = iterator.get_id().to_str();
  // flag for leq to increment symbolic bound
  bool leqBound = false;
  auto upperBound = getUpperBound(forNode, leqBound);
  auto incrementAsInt = std::abs(getIntFromIslExpr(increment));

  auto ctx = MLIRBuilder_.getContext();
  AffineForOp loop;

  if (isInt(lowerBound) && isInt(upperBound)) {
    auto upperBoundAsInt = getIntFromIslExpr(upperBound) + 1;
    auto lowerBoundAsInt = getIntFromIslExpr(lowerBound);
    loop = MLIRBuilder_.createLoop(lowerBoundAsInt, upperBoundAsInt,
                                   incrementAsInt);
  } else if (isInt(lowerBound) && !isInt(upperBound)) {
    auto upperBoundAsExpr = getAffineFromIslExpr(upperBound, ctx);
    std::string upperBoundId = "";
    getBoundId(upperBound, upperBoundId);
    auto lowerBoundAsInt = getIntFromIslExpr(lowerBound);
    loop = MLIRBuilder_.createLoop(lowerBoundAsInt, upperBoundAsExpr,
                                   upperBoundId, incrementAsInt, leqBound);
  } else if (!isInt(lowerBound) && isInt(upperBound)) {
    auto upperBoundAsInt = getIntFromIslExpr(upperBound) + 1;
    auto lowerBoundAsExpr = getAffineFromIslExpr(lowerBound, ctx);
    std::string lowerBoundId = "";
    getBoundId(lowerBound, lowerBoundId);
    loop = MLIRBuilder_.createLoop(lowerBoundAsExpr, lowerBoundId,
                                   upperBoundAsInt, incrementAsInt);
  } else {
    auto upperBoundAsExpr = getAffineFromIslExpr(upperBound, ctx);
    auto lowerBoundAsExpr = getAffineFromIslExpr(lowerBound, ctx);
    std::string upperBoundId = "";
    getBoundId(upperBound, upperBoundId);
    std::string lowerBoundId = "";
    getBoundId(lowerBound, lowerBoundId);
    loop =
        MLIRBuilder_.createLoop(lowerBoundAsExpr, lowerBoundId,
                                upperBoundAsExpr, upperBoundId, incrementAsInt);
  }

  auto resInsertion =
      MLIRBuilder_.getLoopTable().insert(iteratorId, loop.getInductionVar());
  if (failed(resInsertion))
    llvm_unreachable("failed to insert in loop table");

  // create loop body.
  MLIRFromISLAstImpl(forNode.for_get_body());

  // update indvar before erasing if it is visible
  // outside the loop. If it is visible outside the
  // loop, the value will be in the symbol table.
  mlir::Value indVar;
  if (succeeded(MLIRBuilder_.getSymbolInductionVar(iteratorId, indVar))) {
    // TODO: finish me.
  }

  // induction variable goes out of scop. Remove from
  // loopTable
  MLIRBuilder_.getLoopTable().erase(iteratorId);

  // set the insertion point after the loop operation.
  MLIRBuilder_.setInsertionPointAfter(&loop);
}

void IslNodeBuilder::createUser(isl::ast_node userNode) {
  auto expression = userNode.user_get_expr();
  if (isl_ast_expr_get_type(expression.get()) != isl_ast_expr_op)
    llvm_unreachable("expect isl_ast_expr_op");
  if (isl_ast_expr_get_op_type(expression.get()) != isl_ast_op_call)
    llvm_unreachable("expect operation of type call");
  auto stmtExpression = expression.get_op_arg(0);
  auto stmtId = stmtExpression.get_id();
  auto stmt = islAst_.getScop().getStmt(stmtId);
  if (!stmt)
    llvm_unreachable("cannot find statement");
  auto body = stmt->body;
  if (pet_tree_get_type(body) != pet_tree_expr)
    llvm_unreachable("expect pet_tree_expr");
  auto expr = pet_tree_expr_get_expr(body);

  if (failed(MLIRBuilder_.createStmt(expr))) {
    MLIRBuilder_.dump();
    llvm_unreachable("cannot generate statement");
  }
}

void IslNodeBuilder::createTransposeOperation(std::vector<std::string> list) {
  auto vec = MLIRBuilder_.getAccess(list);

  if (failed(MLIRBuilder_.createTransposeOperation(vec[0], vec[1]))) {
    MLIRBuilder_.dump();
    llvm_unreachable("cannot generate blas function");
  }
}
void IslNodeBuilder::createMatVecOperation(std::vector<std::string> list) {
  int rhs = std::stoi(list.back());
  list.pop_back();
  auto vec = MLIRBuilder_.getAccess(list);

  if (failed(MLIRBuilder_.createMatVecOperation(vec[1], vec[2], vec[0], rhs))) {
    MLIRBuilder_.dump();
    llvm_unreachable("cannot generate blas function");
  }
}

void IslNodeBuilder::createBlasOperation(std::vector<std::string> list) {
  /*  // get to the user
  if (isl_ast_node_get_type(markNode.get()) != isl_ast_node_mark)
    llvm_unreachable("code generation error");
  // isl_ast_node_mark_get_node

         auto node = markNode.mark_get_node();
    if (isl_ast_node_get_type(node.get()) != isl_ast_node_for)
      llvm_unreachable("code generation error");
    node = node.for_get_body();
    if (isl_ast_node_get_type(node.get()) != isl_ast_node_for)
      llvm_unreachable("code generation error");
    node = node.for_get_body();
    if (isl_ast_node_get_type(node.get()) != isl_ast_node_for)
      llvm_unreachable("code generation error");
    node = node.for_get_body();
    if (isl_ast_node_get_type(node.get()) != isl_ast_node_user)
      llvm_unreachable("code generation error");

    auto expression = node.user_get_expr();
    if (isl_ast_expr_get_type(expression.get()) != isl_ast_expr_op)
      llvm_unreachable("expect isl_ast_expr_op");
    if (isl_ast_expr_get_op_type(expression.get()) != isl_ast_op_call)
      llvm_unreachable("expect operation of type call");
    auto stmtExpression = expression.get_op_arg(0);
    auto stmtId = stmtExpression.get_id();
    auto stmt = islAst_.getScop().getStmt(stmtId);
    if (!stmt)
      llvm_unreachable("cannot find statement");
    auto body = stmt->body;
    if (pet_tree_get_type(body) != pet_tree_expr)
      llvm_unreachable("expect pet_tree_expr");
    auto expr = pet_tree_expr_get_expr(body);
    /////
        //auto reads = pet_expr_access_get_may_read(expr);
        //auto writes = pet_expr_access_get_may_write(expr);
        //outs() << isl::manage(stmt->domain).get_tuple_name();
        //isl::manage(reads).dump();
        //isl::manage(writes).dump();
      /*reads =
  reads.apply_domain(band.child(0).get_prefix_schedule_union_map()); writes =
  writes.apply_domain(band.child(0).get_prefix_schedule_union_map());
        reads.dump();
        outs() << "\n...\n";
        writes.dump();
        outs() << "\n...\n";
        outs() << "\n...\n";
        reads.range().dump();
        reads.range().foreach_set([&](isl::set s) -> isl_stat {
       outs() <<s.get_tuple_name();
      return isl_stat_ok;
    });



    /////
    // assume 2 args per op expr:
    if (pet_expr_get_n_arg(expr) != 2)
      llvm_unreachable("cannot handle the args");
    */
  auto vec = MLIRBuilder_.getAccess(list);
  // pet_expr_free(expr);

  // if (failed(MLIRBuilder_.createBlasOperation(vec[0], vec[1], vec[3],
  // vec[2]))) {
  if (failed(MLIRBuilder_.createBlasOperation(vec[2], vec[1], vec[0]))) {
    MLIRBuilder_.dump();
    llvm_unreachable("cannot generate blas function");
  }
}

void IslNodeBuilder::createBlock(isl::ast_node blockNode) {
  auto list = blockNode.block_get_children();
  for (int i = 0; i < list.n_ast_node(); i++)
    MLIRFromISLAstImpl(list.get_ast_node(i));
}

void IslNodeBuilder::createIf(isl::ast_node ifNode) {
  outs() << __func__ << "\n";
}

void IslNodeBuilder::MLIRFromISLAstImpl(isl::ast_node node) {
  // std::cout << node.to_str() << "\n";
  switch (isl_ast_node_get_type(node.get())) {
  case isl_ast_node_error:
    llvm_unreachable("code generation error");
  case isl_ast_node_for:
    createFor(node);
    return;
  case isl_ast_node_user:
    createUser(node);
    return;
  case isl_ast_node_block:
    createBlock(node);
    return;
  case isl_ast_node_if:
    createIf(node);
    return;
  case isl_ast_node_mark: {
    auto name = node.mark_get_id().get_name();

    //outs() << name;
    if (!std::string(name).compare(std::string("MatMul"))) {
      std::vector<std::string> *vector;
      vector = (std::vector<std::string> *)(node.mark_get_id().get_user());

      createBlasOperation(*vector);
      delete vector;

    } else if (!std::string(name).compare(std::string("Transpose"))) {

      std::vector<std::string> *vector;
      vector = (std::vector<std::string> *)(node.mark_get_id().get_user());
       for(auto s : *vector){
        // outs() << "\n" << s << "\n";
       }
      createTransposeOperation(*vector);
      delete vector;
      return;

     } else if (!std::string(name).compare(std::string("MatVec"))) {

      std::vector<std::string> *vector;
      vector = (std::vector<std::string> *)(node.mark_get_id().get_user());
       for(auto s : *vector){
        // outs() << "\n" << s << "\n";
       }
      createMatVecOperation(*vector);
      delete vector;
      return;
    } else {
      llvm_unreachable("Mark type not supported");
    }
    return;
  }

  }
  llvm_unreachable("unknown isl_ast_node_type");
}

void IslNodeBuilder::MLIRFromISLAst() {
  isl::ast_node root = islAst_.getRoot();
  MLIRFromISLAstImpl(root);
  // insert return statement.
  MLIRBuilder_.createReturn();
  MLIRBuilder_.runPasses();

  // verify the module after we have finisched constructing it.
  if (failed(MLIRBuilder_.verifyModule()))
    llvm_unreachable("module verification error");
}
