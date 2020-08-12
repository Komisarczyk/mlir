#include "detection.h"
#include "access_patterns.h"
#include "pet.h"

#include "ctx.h"
#include "islAst.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include <fstream>
#include <iostream>

using namespace mlir;
using namespace llvm;
using namespace pet;
#define DEBUG_TYPE "pet-to-mlir-codegen"

isl::schedule_node rebuild(isl::schedule_node node,
                           const builders::ScheduleNodeBuilder &replacement) {
  // this may not be always legal...
  node = node.cut();
  node = replacement.insertAt(node);
  return node;
}
isl::schedule_node
replaceOnce(isl::schedule_node node,
            const matchers::ScheduleNodeMatcher &pattern,
            const builders::ScheduleNodeBuilder &replacement) {
  if (matchers::ScheduleNodeMatcher::isMatching(pattern, node)) {
    LLVM_DEBUG(dbgs() << "match success!\n");
    node = rebuild(node, replacement);
  }
  return node;
}

isl::schedule_node
replaceDFSPreorderOnce(isl::schedule_node node,
                       const matchers::ScheduleNodeMatcher &pattern,
                       const builders::ScheduleNodeBuilder &replacement) {
  node = replaceOnce(node, pattern, replacement);
  if ((isl_schedule_node_get_type(node.get())) == isl_schedule_node_mark) {
    return node;
  }
  for (int i = 0; i < node.n_children(); ++i) {
    node = replaceDFSPreorderOnce(node.child(i), pattern, replacement).parent();
  }
  return node;
}

isl::schedule runDetection(pet::Scop &scop) {
  auto root = scop.getSchedule().get_root();

  isl::schedule_node ijk;

  using namespace matchers;

  // check if the partial schedule is 3d.
  auto is3d = [](isl::schedule_node band) {
    auto umap = band.child(0).get_prefix_schedule_union_map();
    if (umap.n_map() != 1)
      return false;
    auto map = isl::map::from_union_map(umap);

    return map.dim(isl::dim::out) == 3;
  };
  auto is2d = [](isl::schedule_node band) {
    auto umap = band.child(0).get_prefix_schedule_union_map();
    if (umap.n_map() != 1)
      return false;
    auto map = isl::map::from_union_map(umap);
    return map.dim(isl::dim::out) == 2;
  };
  auto hasGemmAccess =
      [&scop](isl::schedule_node band,
              SmallVector<std::vector<std::string> *, 4> *listQueue) {
        auto reads = scop.getReads();
        auto writes = scop.getMustWrites();
        reads =
            reads.apply_domain(band.child(0).get_prefix_schedule_union_map());
        writes =
            writes.apply_domain(band.child(0).get_prefix_schedule_union_map());
        using namespace matchers;
        auto ctx = band.get_ctx();
        auto _i = placeholder(ctx);
        auto _ii = placeholder(ctx);
        auto _j = placeholder(ctx);
        auto _jj = placeholder(ctx);
        auto _k = placeholder(ctx);
        auto _A = arrayPlaceholder();
        auto _B = arrayPlaceholder();
        auto _C = arrayPlaceholder();

        // placeholder are *not* reused across different calls of allOf.
        auto psRead =
            allOf(access(_C, _i, _j), access(_A, _i, _k), access(_B, _k, _j));
        auto psWrite = allOf(access(_B, _ii, _jj));
        auto readMatches = match(reads, psRead);
        auto writeMatches = match(writes, psWrite);

        if ((readMatches.size() != 1) || (writeMatches.size() != 1))
          return false;

        if ((readMatches[0][_i].payload().inputDimPos_ !=
             writeMatches[0][_ii].payload().inputDimPos_) ||
            (readMatches[0][_j].payload().inputDimPos_ !=
             writeMatches[0][_jj].payload().inputDimPos_))
          return false;

        // only allocate when matched TODO: should we check structural match?
        auto vec = readMatches[0][dim(0, _k)].candidateSpaces();

        auto accessList = new std::vector<std::string>;
        writes.foreach_map([&accessList](isl::noexceptions::map m) -> isl_stat {
          accessList->push_back(m.range().get_tuple_id().to_str());
          return isl_stat_ok;
        });
        for (auto space : vec) {
          accessList->push_back(
              space.range().get_tuple_id(isl::noexceptions::dim(3)).to_str());
          // outs() <<
          // space.range().get_tuple_id(isl::noexceptions::dim(3)).to_str();
        }
        /*
        reads.foreach_map([&accessList](isl::noexceptions::map m) -> isl_stat{

          accessList->push_back(m.range().get_tuple_id().to_str());
          return isl_stat_ok;
        });
        */
        listQueue->push_back(accessList);

        return true;
      };

  auto hasTransposeAccess =
      [&scop](isl::schedule_node band,
              SmallVector<std::vector<std::string> *, 4> *listQueue) {
        auto reads = scop.getReads();
        auto writes = scop.getMustWrites();
        reads =
            reads.apply_domain(band.child(0).get_prefix_schedule_union_map());
        writes =
            writes.apply_domain(band.child(0).get_prefix_schedule_union_map());
        auto v = scop.getInputArrays();

        auto ReadsAndWrites = reads.unite(writes);

        using namespace matchers;
        auto ctx = band.get_ctx();
        auto _i = placeholder(ctx);
        auto _j = placeholder(ctx);
        auto _A = arrayPlaceholder();
        auto _B = arrayPlaceholder();

        // placeholder are *not* reused across different calls of allOf.
        auto ps = allOf(access(_A, _i, _j), access(_B, _j, _i));

        auto Matches = match(ReadsAndWrites, ps);

        if (Matches.size() != 2)
          return false;

        auto accessList = new std::vector<std::string>;
        writes.foreach_map([&accessList](isl::noexceptions::map m) -> isl_stat {
          accessList->push_back(m.range().get_tuple_id().to_str());
          return isl_stat_ok;
        });
        reads.foreach_map([&accessList](isl::noexceptions::map m) -> isl_stat {
          accessList->push_back(m.range().get_tuple_id().to_str());
          return isl_stat_ok;
        });
        listQueue->push_back(accessList);

        return true;
      };

  SmallVector<std::vector<std::string> *, 4> listQueue;
  auto isTranspose = [&](isl::schedule_node band) {
    return is2d(band) && hasTransposeAccess(band, &listQueue);
  };

  // auto readList = new SmallVector<std::string,4>;
  auto isGemmLike = [&](isl::schedule_node band) {
    return is3d(band) && hasGemmAccess(band, &listQueue);
  };

  // clang-format off
  auto matcher =
    band(isGemmLike, ijk, 
    leaf());
  // clang-format on

  auto builder = builders::ScheduleNodeBuilder();
  {
    using namespace builders;
    auto scheduleIJK = [&]() { return ijk.band_get_partial_schedule(); };
    auto marker = [&]() {
      auto pointer = listQueue.front();
      listQueue.erase(listQueue.begin());
      return isl::id::alloc(ijk.get_ctx(), "MatMul", pointer);
    };
    // clang-format off
    builder = mark(marker, band(scheduleIJK));
    // clang-format on
  }

  root = replaceDFSPreorderOnce(root, matcher, builder);
  // clean the list for the transpose marker
  listQueue.clear();

  // clang-format off
  auto matcherT =
    band(isTranspose, ijk, 
    leaf());
  // clang-format on

  auto builderT = builders::ScheduleNodeBuilder();
  {
    using namespace builders;
    auto scheduleIJK = [&]() { return ijk.band_get_partial_schedule(); };
    auto marker = [&]() {
      auto pointer = listQueue.front();
      listQueue.erase(listQueue.begin());
      return isl::id::alloc(ijk.get_ctx(), "Transpose", pointer);
    };
    // clang-format off
    builderT = mark(marker, band(scheduleIJK));
    // clang-format on
  }

  root = replaceDFSPreorderOnce(root, matcherT, builderT);

  return root.get_schedule();
}