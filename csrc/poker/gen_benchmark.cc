// Copyright (c) Facebook, Inc. and its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <iostream>
#include <map>
#include <string>

#include <torch/script.h>

#include "rela/context.h"
#include "rela/data_loop.h"
#include "rela/model_locker.h"
#include "rela/prioritized_replay.h"

#include "real_net.h"
#include "recursive_solving.h"
#include "subgame_solving.h"
#include "util.h"

// POKER VARIANT: Updated namespace
using namespace poker;
using namespace rela;

int get_depth(const Tree& tree, int root = 0) {
  int depth = 1;
  for (auto child : ChildrenIt(tree[root])) {
    depth = std::max(depth, 1 + get_depth(tree, child));
  }
  return depth;
}

struct Timer {
  std::chrono::time_point<std::chrono::system_clock> start =
      std::chrono::system_clock::now();

  double tick() {
    const auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
  }
};

// POKER VARIANT: gen_benchmark generates training data for value networks
// This generates (state, value) pairs by solving subgames recursively
// Why not use logs? Logs contain actual game play, but we need:
// 1. Value estimates for ALL possible states (not just played ones)
// 2. Values computed by solving subgames (more accurate than observed outcomes)
// 3. Training data that covers the full state space for better generalization
int main(int argc, char* argv[]) {
  // POKER: Game() takes no parameters, removed num_dice/num_faces
  int fp_iters = 1024;
  int mdp_depth = 2;
  int num_threads = 10;
  int per_gpu = 1;
  int num_cycles = 6;
  std::string device = "cuda:1";
  std::string net_path;
  {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      // POKER: Removed --num_dice and --num_faces flags (not needed)
      if (arg == "--fp_iters") {
        assert(i + 1 < argc);
        fp_iters = std::stoi(argv[++i]);
      } else if (arg == "--mdp_depth") {
        assert(i + 1 < argc);
        mdp_depth = std::stoi(argv[++i]);
      } else if (arg == "--num_threads") {
        assert(i + 1 < argc);
        num_threads = std::stoi(argv[++i]);
      } else if (arg == "--per_gpu") {
        assert(i + 1 < argc);
        per_gpu = std::stoi(argv[++i]);
      } else if (arg == "--num_cycles") {
        assert(i + 1 < argc);
        num_cycles = std::stoi(argv[++i]);
      } else if (arg == "--device") {
        assert(i + 1 < argc);
        device = argv[++i];
      } else if (arg == "--net") {
        assert(i + 1 < argc);
        net_path = argv[++i];
      } else {
        std::cerr << "Unknown flag: " << arg << "\n";
        return -1;
      }
    }
  }
  // POKER: Game() takes no parameters
  assert(mdp_depth != -1);
  assert(mdp_depth > 0);
  assert(!net_path.empty());

  const Game game;  // POKER: No parameters needed
  std::cout << "Poker game: " << game.num_hands() << " possible hands\n";
  {
    const auto full_tree = unroll_tree(game);
    std::cout << "Tree of depth " << get_depth(full_tree) << " has "
              << full_tree.size() << " nodes\n";
  }

  std::vector<TorchJitModel> models;
  for (int i = 0; i < per_gpu; ++i) {
    auto module = torch::jit::load(net_path);
    module.eval();
    module.to(device);
    models.push_back(module);
  }
  std::vector<TorchJitModel*> model_ptrs;
  for (int i = 0; i < per_gpu; ++i) {
    model_ptrs.push_back(&models[i]);
  }
  auto locker = std::make_shared<ModelLocker>(model_ptrs, device);
  auto replay = std::make_shared<ValuePrioritizedReplay>(1 << 20, 1000, 1.0,
                                                         0.4, 3, false, false);
  auto context = std::make_shared<Context>();

  // POKER: RecursiveSolvingParams no longer needs num_dice/num_faces
  RecursiveSolvingParams cfg;
  cfg.subgame_params.num_iters = fp_iters;
  cfg.subgame_params.linear_update = true;
  cfg.subgame_params.optimistic = false;
  cfg.subgame_params.max_depth = mdp_depth;
  for (int i = 0; i < num_threads; ++i) {
    const int seed = i;
    auto connector = std::make_shared<CVNetBufferConnector>(locker, replay);
    std::shared_ptr<ThreadLoop> loop =
        std::make_shared<DataThreadLoop>(std::move(connector), cfg, seed);
    context->pushThreadLoop(loop);
  }
  std::cout << "Starting the context" << std::endl;
  context->start();
  Timer t;
  for (int i = 0; i < num_cycles; ++i) {
    std::this_thread::sleep_for(std::chrono::seconds(10));
    double secs = t.tick();
    auto added = replay->numAdd();
    std::cout << "time=" << secs << " "
              << "items=" << added << " per_second=" << added / secs << "\n";
  }
}