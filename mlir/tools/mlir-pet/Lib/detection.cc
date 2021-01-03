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
isl::noexceptions::schedule_node
detectGEMM(pet::Scop &scop, isl::noexceptions::schedule_node &root,
           SmallVector<std::vector<std::string> *, 4> &listQueue) {
  using namespace matchers;
  isl::schedule_node ijk;
  // check if the partial schedule is 3d.
  auto is3d = [](isl::schedule_node band) {
    auto umap = band.child(0).get_prefix_schedule_union_map();
    if (umap.n_map() != 1)
      return false;
    auto map = isl::map::from_union_map(umap);

    return map.dim(isl::dim::out) == 3;
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

  auto isGemmLike = [&](isl::schedule_node band) {
    return is3d(band) && hasGemmAccess(band, &listQueue);
  };

  // clang-format off
  auto matcher =
    band(isGemmLike, ijk, 
    leaf());

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

  return replaceDFSPreorderOnce(root, matcher, builder);
}
isl::noexceptions::schedule_node
detectTranspose(pet::Scop &scop, isl::noexceptions::schedule_node &root,
                SmallVector<std::vector<std::string> *, 4> &listQueue) {
  using namespace matchers;
  isl::schedule_node ijk;
  auto is2d = [](isl::schedule_node band) {
    auto umap = band.child(0).get_prefix_schedule_union_map();
    if (umap.n_map() != 1)
      return false;
    auto map = isl::map::from_union_map(umap);
    return map.dim(isl::dim::out) == 2;
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
  auto isTranspose = [&](isl::schedule_node band) {
    return is2d(band) && hasTransposeAccess(band, &listQueue);
  };
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

  return replaceDFSPreorderOnce(root, matcherT, builderT);
}
isl::noexceptions::schedule_node
detectMatVec(pet::Scop &scop, isl::noexceptions::schedule_node &root,
             SmallVector<std::vector<std::string> *, 4> &listQueue) {
  using namespace matchers;
  isl::schedule_node ijk;
  // check if the partial schedule is 3d.
  auto is2d = [](isl::schedule_node band) {
    auto umap = band.child(0).get_prefix_schedule_union_map();
    if (umap.n_map() != 1)
      return false;
    auto map = isl::map::from_union_map(umap);
    return map.dim(isl::dim::out) == 2;
  };
  auto hasMatVecAccess =
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
        auto _A = arrayPlaceholder();
        auto _x = arrayPlaceholder();
        auto _y = arrayPlaceholder();

        // placeholder are *not* reused across different calls of allOf.
        auto psRead = allOf(access(_A, _i, _j), access(_x, _i), access(_y, _j));
        auto psWrite = allOf(access(_x, _ii));
        auto readMatches = match(reads, psRead);
        auto writeMatches = match(writes, psWrite);
        
        if ((readMatches.size() != 1) || (writeMatches.size() != 1))
          return false;

        if (!((readMatches[0][_i].payload().inputDimPos_ !=
               writeMatches[0][_ii].payload().inputDimPos_) !=
              (readMatches[0][_j].payload().inputDimPos_ !=
               writeMatches[0][_ii].payload().inputDimPos_)))
          return false;

        // only allocate when matched
        // TODO:is order of A and y always same from candidateSpaces?

        std::vector<isl::noexceptions::space> vec;
        std::string rhs;
        if ((readMatches[0][_i].payload().inputDimPos_ !=
             writeMatches[0][_ii].payload().inputDimPos_)){
          vec = readMatches[0][dim(0, _i)].candidateSpaces();
          //outs() << "\n" << readMatches[0][_j].payload().inputDimPos_;
           rhs = "0";
        }
        else{
          vec = readMatches[0][dim(0, _j)].candidateSpaces();
          rhs = "1";
          //outs() << "\n" << readMatches[0][_j].payload().inputDimPos_;
        }
        auto accessList = new std::vector<std::string>;
        writes.foreach_map([&accessList](isl::noexceptions::map m) -> isl_stat {
          accessList->push_back(m.range().get_tuple_id().to_str());
          return isl_stat_ok;
        });
        for (auto space : vec) {
          accessList->push_back(
              space.range().get_tuple_id(isl::noexceptions::dim(3)).to_str());
          
        }
        /*
        reads.foreach_map([&accessList](isl::noexceptions::map m) -> isl_stat{

          accessList->push_back(m.range().get_tuple_id().to_str());
          return isl_stat_ok;
        });
        */
       accessList->push_back(rhs);
        listQueue->push_back(accessList);

        return true;
      };

  auto isMatVecLike = [&](isl::schedule_node band) {
    return is2d(band) && hasMatVecAccess(band, &listQueue);
  };

  // clang-format off
  auto matcher =
    band(isMatVecLike, ijk, 
    leaf());

  auto builder = builders::ScheduleNodeBuilder();
  {
    using namespace builders;
    auto scheduleIJK = [&]() { return ijk.band_get_partial_schedule(); };
    auto marker = [&]() {
      auto pointer = listQueue.front();
      listQueue.erase(listQueue.begin());
      return isl::id::alloc(ijk.get_ctx(), "MatVec", pointer);
    };
    // clang-format off
    builder = mark(marker, band(scheduleIJK));
    // clang-format on
  }

  return replaceDFSPreorderOnce(root, matcher, builder);
}
isl::noexceptions::schedule_node
detectSyrk(pet::Scop &scop, isl::noexceptions::schedule_node &root,
           SmallVector<std::vector<std::string> *, 4> &listQueue) {
  using namespace matchers;
  isl::schedule_node ijk;
  // check if the partial schedule is 3d.
  auto is3d = [](isl::schedule_node band) {
    auto umap = band.child(0).get_prefix_schedule_union_map();
    if (umap.n_map() != 1)
      return false;
    auto map = isl::map::from_union_map(umap);
    return map.dim(isl::dim::out) == 3;
  };
 auto hasSyrkAccess =
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
            allOf(access(_B, _i, _j), access(_A, _k, _i), access(_B, _k, _j));
        auto psWrite = allOf(access(_B, _ii, _jj));
        auto readMatches = match(reads, psRead);
        auto writeMatches = match(writes, psWrite);
        outs() << "trying trmm";
        outs() << readMatches.size();
        reads.dump();
        puts("");

        auto dom = reads.domain();
        isl::noexceptions::union_set range;
        isl::noexceptions::set set;
        
        reads.foreach_map([&](isl::noexceptions::map m) -> isl_stat{
          m.dump();
          return isl_stat_ok;
        });
/*
        reads.range().foreach_set([&](isl::noexceptions::set s) -> isl_stat{
          s.foreach_basic_set([&](isl::noexceptions::set ss) -> isl_stat{
            ss.dump();
            set.unite(ss).dump();
            return isl_stat_ok;});
          return isl_stat_ok;
        });
        set.dump();*/
        if ((readMatches.size() != 1) || (writeMatches.size() != 1))
          return false;
 outs() << "trying trmm";
        if ((readMatches[0][_i].payload().inputDimPos_ !=
             writeMatches[0][_ii].payload().inputDimPos_) ||
            (readMatches[0][_j].payload().inputDimPos_ !=
             writeMatches[0][_jj].payload().inputDimPos_))
          return false;
 outs() << "trying trmm";
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

  auto isSyrkLike = [&](isl::schedule_node band) {
    return is3d(band) && hasSyrkAccess(band, &listQueue);
  };

  // clang-format off
  auto matcher =
    band(isSyrkLike, ijk, 
    leaf());

  auto builder = builders::ScheduleNodeBuilder();
  {
    using namespace builders;
    auto scheduleIJK = [&]() { return ijk.band_get_partial_schedule(); };
    auto marker = [&]() {
      auto pointer = listQueue.front();
      listQueue.erase(listQueue.begin());
      return isl::id::alloc(ijk.get_ctx(), "Syrk", pointer);
    };
    // clang-format off
    builder = mark(marker, band(scheduleIJK));
    // clang-format on
  }

  return replaceDFSPreorderOnce(root, matcher, builder);
}



isl::noexceptions::schedule_node
detectDot(pet::Scop &scop, isl::noexceptions::schedule_node &root,
           SmallVector<std::vector<std::string> *, 4> &listQueue) {
  using namespace matchers;
  isl::schedule_node ijk;
  // check if the partial schedule is 3d.
  auto is3d = [](isl::schedule_node band) {
    auto umap = band.child(0).get_prefix_schedule_union_map();
    if (umap.n_map() != 1)
      return false;
    auto map = isl::map::from_union_map(umap);

    return map.dim(isl::dim::out) == 3;
  };
  auto hasDotAccess =
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

  auto isDotLike = [&](isl::schedule_node band) {
    return is3d(band) && hasDotAccess(band, &listQueue);
  };

  // clang-format off
  auto matcher =
    band(isDotLike, ijk, 
    leaf());

  auto builder = builders::ScheduleNodeBuilder();
  {
    using namespace builders;
    auto scheduleIJK = [&]() { return ijk.band_get_partial_schedule(); };
    auto marker = [&]() {
      auto pointer = listQueue.front();
      listQueue.erase(listQueue.begin());
      return isl::id::alloc(ijk.get_ctx(), "Dot", pointer);
    };
    // clang-format off
    builder = mark(marker, band(scheduleIJK));
    // clang-format on
  }

  return replaceDFSPreorderOnce(root, matcher, builder);
}







isl::schedule runDetection(pet::Scop &scop) {
  auto root = scop.getSchedule().get_root();

  using namespace matchers;

  SmallVector<std::vector<std::string> *, 4> listQueue;

  root = detectGEMM(scop, root, listQueue);
  // clang-format on

  // clean the list for the transpose marker
  listQueue.clear();
  root = detectTranspose(scop, root, listQueue);
  listQueue.clear();
  root = detectMatVec(scop, root, listQueue);
  
  //listQueue.clear();
  //root = detectSyrk(scop, root, listQueue);
  return root.get_schedule();
}