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

#include "subgame_solving.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

// POKER VARIANT: Updated include
#include "poker.h"
#include "net_interface.h"
#include "real_net.h"
#include "util.h"

namespace poker {

namespace {

template <class T>
void init_nd(int a, int b, T value, std::vector<std::vector<T>>* array) {
  array->resize(a);
  for (int i = 0; i < a; ++i) {
    (*array)[i].assign(b, value);
  }
}

template <class T>
void init_nd(int a, int b, int c, T value,
             std::vector<std::vector<std::vector<T>>>* array) {
  array->resize(a);
  for (auto& subarray : *array) {
    init_nd(b, c, value, &subarray);
  }
}

// For each node `x` and hand `h` computes
// P(root->x, h | beliefs) := pi^{player}(root->x|h) * P(h).
void compute_reach_probabilities(
    const Tree& tree, const TreeStrategy& strategy,
    const std::vector<double>& initial_beliefs, int player,
    std::vector<std::vector<double>>* reach_probabilities) {
  const auto num_hands = initial_beliefs.size();
  for (size_t node_id = 0; node_id < tree.size(); ++node_id) {
    if (node_id == 0) {
      (*reach_probabilities)[node_id] = initial_beliefs;
    } else {
      const auto& node = tree[node_id];
      const auto& state = node.state;
      // POKER: Get active player from button
      const auto last_action_player_id = Game::get_active_player(tree[node.parent].state);
      const Action last_action = state.last_action;
      if (player == last_action_player_id) {
        for (size_t hand = 0; hand < num_hands; ++hand) {
          (*reach_probabilities)[node_id][hand] =
              (*reach_probabilities)[node.parent][hand] *
              strategy[node.parent][hand][last_action];
        }
      } else {
        (*reach_probabilities)[node_id] = (*reach_probabilities)[node.parent];
      }
    }
  }
}

// POKER VARIANT: Updated to pass state for board card access
std::vector<double> compute_expected_terminal_values(
    const Game& game, const PartialPublicState& state, bool inverse,
    std::vector<double>& op_reach_probabilities) {
  auto values = compute_win_probability(game, state, op_reach_probabilities);
  // Need to convert the probabilities to the payoff the traverser. Note,
  // the probabilities are true probabilities iff op_beliefs sum to 1.
  const auto belief_sum = vector_sum(op_reach_probabilities);
  // Payoff: (probability(win) * 1.0 + probability(lose) * (-1.0).
  for (double& v : values) {
    // v <- ((v / belief_sum) * 2 - 1) * belief_sum;
    v = v * 2 - belief_sum;
  }
  if (inverse) {
    for (double& v : values) {
      v *= -1.0;
    }
  }
  return values;
}

// POKER VARIANT: Query size includes board cards and discard info
// POKER VARIANT: Query size includes all state information
size_t get_query_size(const Game& game) {
  // Format: active_player (1) + traverser (1) + action_one_hot (num_actions) + 
  //         board_cards (6) + discard_choices (2) + street (1) + beliefs (num_hands * 2)
  // SOLVER: Removed pips/stacks/button - not needed for value prediction
  return 1 + 1 + game.num_actions() + 6 + 2 + 1 + game.num_hands() * 2;
}

// POKER VARIANT: Updated to include board cards and discard info in query
int64_t write_query_to(const Game& game, int traverser,
                       const PartialPublicState& state,
                       const std::vector<double>& reaches1,
                       const std::vector<double>& reaches2, float* buffer) {
  int64_t write_index = 0;
  // SOLVER: Encode active player and traverser
  buffer[write_index++] = static_cast<float>(state.player_id);
  buffer[write_index++] = static_cast<float>(traverser);
  
  // Encode last action (one-hot)
  for (Action action = 0; action < game.num_actions(); ++action) {
    buffer[write_index++] = static_cast<float>(action == state.last_action);
  }
  
  // POKER: Encode board cards (6 cards, -1 if not present)
  for (int i = 0; i < 6; ++i) {
    if (i < state.num_board_cards && state.board_cards[i] >= 0) {
      buffer[write_index++] = static_cast<float>(state.board_cards[i]);
    } else {
      buffer[write_index++] = -1.0f;  // No card
    }
  }
  
  // POKER: Encode discard choices (which card each player discarded, -1 if not discarded)
  buffer[write_index++] = static_cast<float>(state.discard_choice[0]);
  buffer[write_index++] = static_cast<float>(state.discard_choice[1]);
  
  // SOLVER: Encode street (removed pips/stacks/button)
  buffer[write_index++] = static_cast<float>(state.street);
  
  // Encode beliefs (normalized probabilities)
  normalize_probabilities_safe(reaches1, kReachSmoothingEps,
                               &buffer[write_index]);
  write_index += reaches1.size();
  normalize_probabilities_safe(reaches2, kReachSmoothingEps,
                               &buffer[write_index]);
  write_index += reaches2.size();
  return write_index;
}

TreeStrategy get_uniform_reach_weigted_strategy(
    const Game& game, const Tree& tree,
    const Pair<std::vector<double>>& initial_beliefs) {
  TreeStrategy strategy = get_uniform_strategy(game, tree);
  std::vector<std::vector<double>> reach_probabilities_buffer;
  init_nd(tree.size(), game.num_hands(), 0.0, &reach_probabilities_buffer);
  for (int traverser : {0, 1}) {
    compute_reach_probabilities(tree, strategy, initial_beliefs[traverser],
                                traverser, &reach_probabilities_buffer);
    for (size_t node = 0; node < tree.size(); ++node) {
      if (!tree[node].num_children() ||
          Game::get_active_player(tree[node].state) != traverser) {
        continue;
      }
      const auto [action_begin, action_end] =
          game.get_bid_range(tree[node].state);
      for (int i = 0; i < game.num_hands(); i++) {
        for (Action a = action_begin; a < action_end; ++a) {
          strategy[node][i][a] *= reach_probabilities_buffer[node][i];
        }
      }
    }
  }
  return strategy;
}

// Helper base class for tree traversing.
struct PartialTreeTraverser {
  const Game game;
  const Tree tree;

  // Probability to reach a specific node by a player with specific under the
  // average policy: [2, num_nodes, num_hands].
  // Computed with precompute_reaches.
  Pair<std::vector<std::vector<double>>> reach_probabilities;

  // Values for each node and hand for one of the players.
  // Shape [num_nodes, num_hands].
  // Leaf values could be populated with precompute_leaf_values.
  // It's up to subclasess to pupulate the rest.
  std::vector<std::vector<double>> traverser_values;

  // Size of the inputs and outputs of the value network.
  const int64_t query_size, output_size;

  PartialTreeTraverser(const Game& game, const Tree& tree,
                       std::shared_ptr<IValueNet> value_net)
      : game(game),
        tree(tree),
        query_size(get_query_size(game)),
        output_size(game.num_hands()),
        value_net(value_net) {
    if (value_net == nullptr) {
      // Check all leaf nodes are final.
      for (auto& node : tree) {
        if (!game.is_terminal(node.state) && !node.num_children()) {
          throw std::runtime_error("Found a node " +
                                   game.state_to_string(node.state) +
                                   " that is a non-final leaf. Either provide "
                                   "value net or increase max_depth");
        }
      }
    } else {
      // Initialzer buffers to query the neural network.
      for (size_t node_id = 0; node_id < tree.size(); ++node_id) {
        const auto& node = tree[node_id];
        const auto& state = node.state;
        if (!node.num_children() && !game.is_terminal(state)) {
          pseudo_leaves_indices.push_back(node_id);
        }
      }
      net_query_buffer.resize(query_size * pseudo_leaves_indices.size());
    }
    for (size_t i = 0; i < tree.size(); ++i) {
      if (game.is_terminal(tree[i].state)) {
        terminal_indices.push_back(i);
      }
    }
    leaf_values =
        torch::empty({(int64_t)pseudo_leaves_indices.size(), output_size});
    init_nd(tree.size(), game.num_hands(), 0.0, &traverser_values);
    init_nd(tree.size(), game.num_hands(), 0.0, &reach_probabilities[0]);
    init_nd(tree.size(), game.num_hands(), 0.0, &reach_probabilities[1]);
  }

  // Write a single query to the buffer. The query corresponds to the node as
  // seen by tranverser.
  void write_query(size_t node_id, int traverser, float* buffer) {
    const auto& state = tree[node_id].state;
    auto write_index =
        write_query_to(game, traverser, state, reach_probabilities[0][node_id],
                       reach_probabilities[1][node_id], buffer);
    assert(write_index == query_size);
  }

  void add_training_example(int traverser, const std::vector<double>& values) {
    auto query_tensor = torch::zeros({1, query_size});
    auto value_tensor = torch::zeros({1, output_size});
    write_query(/*node_id=*/0, traverser, query_tensor.data_ptr<float>());
    std::copy_n(values.begin(), output_size, value_tensor.data_ptr<float>());
    value_net->add_training_example(query_tensor, value_tensor);
  }

  void precompute_reaches(const TreeStrategy& strategy,
                          const std::vector<double>& initial_beliefs,
                          int player) {
    poker::compute_reach_probabilities(
        tree, strategy, initial_beliefs, player, &reach_probabilities[player]);
  }

  // Compute values for leaf nodes. For terminals exact value is used; for
  // non-terminals value net is called. Reaches for both players must be
  // precomputed.
  void precompute_all_leaf_values(int traverser) {
    query_value_net(traverser);
    populate_leaf_values();
    precompute_terminal_leaves_values(traverser);
  }

 protected:
  void precompute_reaches(const TreeStrategy& strategy,
                          const Pair<std::vector<double>>& initial_beliefs) {
    precompute_reaches(strategy, initial_beliefs[0], 0);
    precompute_reaches(strategy, initial_beliefs[1], 1);
  }

  // Query value net, weight by oponent reaches, and save result as
  // leaf_values tensor.
  void query_value_net(int traverser) {
    if (pseudo_leaves_indices.empty()) return;
    assert(value_net != nullptr);
    const int64_t N = pseudo_leaves_indices.size();
    torch::Tensor scalers = torch::zeros({N}, torch::kDouble);
    auto scalers_acc = scalers.accessor<double, 1>();
    for (size_t row = 0; row < pseudo_leaves_indices.size(); ++row) {
      const auto node_id = pseudo_leaves_indices[row];
      write_query(node_id, traverser,
                  net_query_buffer.data() + row * query_size);
      scalers_acc[row] =
          vector_sum(reach_probabilities[1 - traverser][node_id]);
    }
    leaf_values = value_net->compute_values(
        torch::from_blob(net_query_buffer.data(), {N, query_size}));
    leaf_values *= scalers.unsqueeze(1);
  }

  // Copy results from leaf_values to corresponding nodes in
  // traverser_values.
  void populate_leaf_values() {
    if (pseudo_leaves_indices.empty()) return;
    auto result_acc = leaf_values.accessor<float, 2>();
    for (size_t row = 0; row < pseudo_leaves_indices.size(); ++row) {
      const auto node_id = pseudo_leaves_indices[row];
      for (int64_t i = 0; i < output_size; ++i) {
        traverser_values[node_id][i] = result_acc[row][i];
      }
    }
  }

  // Populate traverser_values for terminal nodes.
  void precompute_terminal_leaves_values(int traverser) {
    // POKER VARIANT: Pass state instead of last_bid for board card access
    for (auto node_id : terminal_indices) {
      traverser_values[node_id] = compute_expected_terminal_values(
          game, tree[node_id].state,
          /*inverse=*/game.get_active_player(tree[node_id].state) != traverser,
          reach_probabilities[1 - traverser][node_id]);
    }
  }

  // List of pseude leaf nodes, i.e., nodes where value net eval is needed.
  std::vector<size_t> pseudo_leaves_indices;
  std::vector<size_t> terminal_indices;
  // Query buffers.
  std::vector<float> net_query_buffer;
  torch::Tensor leaf_values;

  std::shared_ptr<IValueNet> value_net;
};

struct BRSolver : public PartialTreeTraverser {
  BRSolver(const Game& game, const std::vector<UnrolledTreeNode>& tree,
           std::shared_ptr<IValueNet> value_net)
      : PartialTreeTraverser(game, tree, value_net) {
    init_nd(tree.size(), game.num_hands(), game.num_actions(), 0.0,
            &br_strategies);
  }

  // Re-computes BR strategy for the traverser and returns its expected BR
  // value and the best response strategy. Only values for nodes where
  // traverser is acting are valid.
  const TreeStrategy& compute_br(
      int traverser, const TreeStrategy& oponent_strategy,
      const Pair<std::vector<double>>& initial_beliefs,
      std::vector<double>* values) {
    precompute_reaches(oponent_strategy, initial_beliefs);
    precompute_all_leaf_values(traverser);
    for (size_t public_node = tree.size(); public_node-- > 0;) {
      const auto& node = tree[public_node];
      auto& value = traverser_values[public_node];
      if (!node.num_children()) {
        // All leaf values are set by precompute_all_leaf_values.
        continue;
      }
      const auto& state = node.state;
      value.assign(value.size(), 0.0);
      if (Game::get_active_player(state) == traverser) {
        std::vector<int> best_action(game.num_hands());
        for (auto [child_node, action] : ChildrenActionIt(node, game)) {
          const auto& new_value = traverser_values[child_node];
          for (int hand = 0; hand < game.num_hands(); ++hand) {
            if (child_node == node.children_begin ||
                new_value[hand] > value[hand]) {
              value[hand] = new_value[hand];
              best_action[hand] = action;
            }
          }
        }
        for (int hand = 0; hand < game.num_hands(); ++hand) {
          br_strategies[public_node][hand].assign(game.num_actions(), 0.);
          br_strategies[public_node][hand][best_action[hand]] = 1.0;
        }
      } else {
        for (auto child_node : ChildrenIt(node)) {
          const auto& new_value = traverser_values[child_node];
          for (int hand = 0; hand < game.num_hands(); ++hand) {
            value[hand] += new_value[hand];
          }
        }
      }
    }
    *values = traverser_values[0];
    return br_strategies;
  }

  // Indexed by [node, hand, action].
  TreeStrategy br_strategies;
};

struct FP : public ISubgameSolver {
  FP(const Game& game, const Tree& tree, std::shared_ptr<IValueNet> value_net,
     const Pair<std::vector<double>>& beliefs,
     const SubgameSolvingParams& params)
      : params(params),
        game(game),
        num_strategies(0),
        // TODO(akhti): normalize before using!
        initial_beliefs(beliefs),
        tree(tree),
        br_solver(game, tree, value_net) {
    // Initial strategies are uniform over feasible actions.
    average_strategies = get_uniform_strategy(game, tree);
    last_strategies = average_strategies;
    sum_strategies =
        get_uniform_reach_weigted_strategy(game, tree, initial_beliefs);
    assert(!params.use_cfr);
  }

  FP(const Game& game, const PartialPublicState& root,
     std::shared_ptr<IValueNet> value_net,
     const Pair<std::vector<double>>& beliefs,
     const SubgameSolvingParams& params)
      : FP(game, unroll_tree(game, root, params.max_depth), value_net, beliefs,
           params) {}

  void update_sum_strat(int public_node, int traverser,
                        const TreeStrategy& br_strategies,
                        const std::vector<double>& traverser_beliefs) {
    const auto& node = tree[public_node];
    const auto& state = node.state;
    if (node.num_children()) {
      if (Game::get_active_player(state) == traverser) {
        std::vector<double> new_beliefs(game.num_hands());
        for (auto [child_node, a] : ChildrenActionIt(node, game)) {
          for (int i = 0; i < game.num_hands(); i++) {
            sum_strategies[public_node][i][a] +=
                traverser_beliefs[i] * br_strategies[public_node][i][a];
            last_strategies[public_node][i][a] =
                traverser_beliefs[i] * br_strategies[public_node][i][a];
          }
          for (int i = 0; i < game.num_hands(); i++) {
            new_beliefs[i] =
                traverser_beliefs[i] * br_strategies[public_node][i][a];
          }
          update_sum_strat(child_node, traverser, br_strategies, new_beliefs);
        }
      } else {
        assert(Game::get_active_player(state) == 1 - traverser);
        for (auto child_node : ChildrenIt(node)) {
          update_sum_strat(child_node, traverser, br_strategies,
                           traverser_beliefs);
        }
      }
    }
  }

  void step(int traverser) override {
    const TreeStrategy& br_strategy =
        br_solver.compute_br(traverser, average_strategies, initial_beliefs,
                             &root_values[traverser]);

    // How many updates done for the valeus and strategy of the traverser
    // assuming alternating pattern.
    const int num_update = num_strategies / 2 + 1;
    {
      const double alpha =
          params.linear_update ? 2. / (num_update + 1) : 1. / (num_update);
      root_values_means[traverser].resize(root_values[traverser].size());
      for (size_t i = 0; i < root_values[traverser].size(); ++i) {
        root_values_means[traverser][i] +=
            (root_values[traverser][i] - root_values_means[traverser][i]) *
            alpha;
      }
    }
    update_sum_strat(/*public_node=*/0, traverser, br_strategy,
                     initial_beliefs[traverser]);
    for (size_t node = 0; node < tree.size(); ++node) {
      if (!tree[node].num_children() ||
          Game::get_active_player(tree[node].state) != traverser) {
        continue;
      }
      for (int i = 0; i < game.num_hands(); i++) {
        if (params.linear_update) {
          for (auto& v : sum_strategies[node][i]) {
            v *= static_cast<double>(num_update + 1) / (num_update + 2);
          }
        }
        if (params.optimistic) {
          // Use safe normalization to prevent assertion failures when strategies become all zeros
          std::vector<double> combined(sum_strategies[node][i].size());
          double combined_sum = 0.0;
          for (size_t j = 0; j < combined.size(); ++j) {
            combined[j] = sum_strategies[node][i][j] + last_strategies[node][i][j];
            combined_sum += combined[j];
          }
          if (combined_sum < kRegretSmoothingEps) {
            std::cerr << "[DEBUG] subgame_solving.cc:480 optimistic: combined_sum=" << combined_sum 
                      << " node=" << node << " hand=" << i << " size=" << combined.size() << std::endl;
          }
          normalize_probabilities_safe(combined, kRegretSmoothingEps,
                                       &average_strategies[node][i]);
        } else {
          double sum_sum = 0.0;
          for (size_t j = 0; j < sum_strategies[node][i].size(); ++j) {
            sum_sum += sum_strategies[node][i][j];
          }
          if (sum_sum < kRegretSmoothingEps) {
            std::cerr << "[DEBUG] subgame_solving.cc:489 non-optimistic: sum_sum=" << sum_sum 
                      << " node=" << node << " hand=" << i << " size=" << sum_strategies[node][i].size() << std::endl;
          }
          normalize_probabilities_safe(sum_strategies[node][i], kRegretSmoothingEps,
                                      &average_strategies[node][i]);
        }
      }
    }
    ++num_strategies;
  }

  void multistep() override {
    for (int iter = 0; iter < params.num_iters; ++iter) {
      step(iter % 2);
    }
  }

  void update_value_network() override {
    br_solver.add_training_example(0, get_hand_values(0));
    br_solver.add_training_example(1, get_hand_values(1));
  }

  const TreeStrategy& get_strategy() const override {
    return average_strategies;
  }

  void print_strategy(const std::string& path) const override {
    poker::print_strategy(game, tree, average_strategies, path);
  }

  std::vector<double> get_hand_values(int player_id) const override {
    assert(num_strategies >= 2);
    return root_values_means.at(player_id);
  }

  const Tree& get_tree() const override { return tree; }

 private:
  const SubgameSolvingParams params;
  const Game game;
  // Num updates accumulated in sum_strategies.
  int num_strategies;
  // Believes for both players: [2, num_hands].
  const Pair<std::vector<double>> initial_beliefs;
  // Indexed by [node, hand, action].
  TreeStrategy average_strategies, sum_strategies, last_strategies;
  // Values from the last traversal at the root: [2, num_hands].
  Pair<std::vector<double>> root_values;
  Pair<std::vector<double>> root_values_means;

  Tree tree;
  BRSolver br_solver;
};

struct CFR : public ISubgameSolver, private PartialTreeTraverser {
  CFR(const Game& game, const Tree& tree, std::shared_ptr<IValueNet> value_net,
      const Pair<std::vector<double>>& beliefs,
      const SubgameSolvingParams& params)
      : PartialTreeTraverser(game, tree, value_net),
        params(params),
        num_steps{0, 0},
        // TODO(akhti): normalize before using!
        initial_beliefs(beliefs) {
    // Initial strategies are uniform over feasible actions.
    average_strategies = get_uniform_strategy(game, tree);
    last_strategies = average_strategies;
    sum_strategies =
        get_uniform_reach_weigted_strategy(game, tree, initial_beliefs);
    init_nd(tree.size(), game.num_hands(), game.num_actions(), 0.0, &regrets);
    init_nd(tree.size(), game.num_hands(), 0.0, &reach_probabilities_buffer);
  }

  CFR(const Game& game, const PartialPublicState& root,
      std::shared_ptr<IValueNet> value_net,
      const Pair<std::vector<double>>& beliefs,
      const SubgameSolvingParams& params)
      : CFR(game, unroll_tree(game, root, params.max_depth), value_net, beliefs,
            params) {
    assert(params.use_cfr);
    assert(!params.linear_update || !params.dcfr);
  }

  // Adds regrets for the last_strategies to regrets.
  // Sets traverser_values[node] to the EVs of last_strategies for traverser.
  void update_regrets(int traverser) {
    precompute_reaches(last_strategies, initial_beliefs);
    precompute_all_leaf_values(traverser);

    for (size_t public_node = tree.size(); public_node-- > 0;) {
      const auto& node = tree[public_node];
      if (!node.num_children()) {
        // All leaf values are set by precompute_all_leaf_values.
        continue;
      }
      const auto& state = node.state;
      auto& value = traverser_values[public_node];
      value.assign(value.size(), 0.0);
      if (Game::get_active_player(state) == traverser) {
        for (auto [child_node, action] : ChildrenActionIt(node, game)) {
          const auto& action_value = traverser_values[child_node];
          for (int hand = 0; hand < game.num_hands(); ++hand) {
            regrets[public_node][hand][action] += action_value[hand];
            value[hand] +=
                action_value[hand] * last_strategies[public_node][hand][action];
          }
        }
        for (int hand = 0; hand < game.num_hands(); ++hand) {
          for (auto [child_node, action] : ChildrenActionIt(node, game)) {
            regrets[public_node][hand][action] -= value[hand];
          }
        }
      } else {
        assert(Game::get_active_player(state) == 1 - traverser);
        for (auto child_node : ChildrenIt(node)) {
          const auto& action_value = traverser_values[child_node];
          for (int hand = 0; hand < game.num_hands(); ++hand) {
            value[hand] += action_value[hand];
          }
        }
      }
    }
  }

  void step(int traverser) override {
    update_regrets(traverser);
    root_values[traverser] = traverser_values[0];
    {
      const double alpha = params.linear_update
                               ? 2. / (num_steps[traverser] + 2)
                               : 1. / (num_steps[traverser] + 1);
      root_values_means[traverser].resize(root_values[traverser].size());
      for (size_t i = 0; i < root_values[traverser].size(); ++i) {
        root_values_means[traverser][i] +=
            (root_values[traverser][i] - root_values_means[traverser][i]) *
            alpha;
      }
    }

    double pos_discount = 1;
    double neg_discount = 1;
    double strat_discount = 1;
    {
      // We always have uniform strategy, hence +1.
      const double num_strategies = num_steps[traverser] + 1;
      if (params.linear_update) {
        pos_discount = neg_discount = strat_discount =
            num_strategies / (num_strategies + 1);
      } else if (params.dcfr) {
        if (params.dcfr_alpha >= 5) {
          pos_discount = 1;
        } else {
          pos_discount = pow(num_strategies, params.dcfr_alpha) /
                         (pow(num_strategies, params.dcfr_alpha) + 1.);
        }
        if (params.dcfr_beta <= -5) {
          neg_discount = 0;
        } else {
          neg_discount = pow(num_strategies, params.dcfr_beta) /
                         (pow(num_strategies, params.dcfr_beta) + 1.);
        }
        strat_discount =
            pow(num_strategies / (num_strategies + 1), params.dcfr_gamma);
      }
    }

    for (size_t node = 0; node < tree.size(); ++node) {
      if (!tree[node].num_children() ||
          Game::get_active_player(tree[node].state) != traverser) {
        continue;
      }
      const auto [start, end] = game.get_bid_range(tree[node].state);
      for (int i = 0; i < game.num_hands(); i++) {
        for (int action = start; action < end; ++action) {
          // TODO(akhti): remove magic constant.
          last_strategies[node][i][action] =
              std::max<double>(regrets[node][i][action], kRegretSmoothingEps);
        }
        // Use safe normalization to prevent assertion failures
        double last_sum = 0.0;
        for (size_t j = 0; j < last_strategies[node][i].size(); ++j) {
          last_sum += last_strategies[node][i][j];
        }
        if (last_sum < kRegretSmoothingEps) {
          std::cerr << "[DEBUG] subgame_solving.cc:679 last_strategies: sum=" << last_sum 
                    << " node=" << node << " hand=" << i << " size=" << last_strategies[node][i].size() << std::endl;
        }
        normalize_probabilities_safe(last_strategies[node][i], kRegretSmoothingEps,
                                     &last_strategies[node][i]);
      }
    }

    compute_reach_probabilities(tree, last_strategies,
                                initial_beliefs[traverser], traverser,
                                &reach_probabilities_buffer);
    for (size_t node = 0; node < tree.size(); ++node) {
      if (!tree[node].num_children() ||
          Game::get_active_player(tree[node].state) != traverser) {
        continue;
      }
      const auto [action_begin, action_end] =
          game.get_bid_range(tree[node].state);
      for (int i = 0; i < game.num_hands(); i++) {
        for (Action a = action_begin; a < action_end; ++a) {
          regrets[node][i][a] *=
              regrets[node][i][a] > 0 ? pos_discount : neg_discount;
        }
        for (Action a = action_begin; a < action_end; ++a) {
          sum_strategies[node][i][a] *= strat_discount;
        }
        for (Action a = action_begin; a < action_end; ++a) {
          sum_strategies[node][i][a] +=
              reach_probabilities_buffer[node][i] * last_strategies[node][i][a];
        }
        // Use safe normalization to prevent assertion failures
        double final_sum = 0.0;
        for (size_t j = 0; j < sum_strategies[node][i].size(); ++j) {
          final_sum += sum_strategies[node][i][j];
        }
        if (final_sum < kRegretSmoothingEps) {
          std::cerr << "[DEBUG] subgame_solving.cc:692 final sum_strategies: sum=" << final_sum 
                    << " node=" << node << " hand=" << i << " size=" << sum_strategies[node][i].size() << std::endl;
        }
        normalize_probabilities_safe(sum_strategies[node][i], kRegretSmoothingEps,
                                     &average_strategies[node][i]);
      }
    }

    ++num_steps[traverser];
  }

  void multistep() override {
    for (int iter = 0; iter < params.num_iters; ++iter) {
      step(iter % 2);
    }
  }

  void update_value_network() override {
    assert(num_steps[0] > 0 && num_steps[1] > 0);
    add_training_example(0, get_hand_values(0));
    add_training_example(1, get_hand_values(1));
  }

  const TreeStrategy& get_strategy() const override {
    return average_strategies;
  }

  const TreeStrategy& get_sampling_strategy() const override {
    return last_strategies;
  }

  const TreeStrategy& get_belief_propogation_strategy() const override {
    return last_strategies;
  }

  void print_strategy(const std::string& path) const override {
    poker::print_strategy(game, tree, average_strategies, path);
  }

  std::vector<double> get_hand_values(int player_id) const override {
    return root_values_means.at(player_id);
  }

  const Tree& get_tree() const override { return tree; }

 private:
  const SubgameSolvingParams params;
  // Num step() done for the player.
  Pair<int> num_steps;
  // Believes for both players: [2, num_hands].
  const Pair<std::vector<double>> initial_beliefs;
  // Indexed by [node, hand, action].
  TreeStrategy average_strategies, sum_strategies, last_strategies;
  TreeStrategy regrets;
  // Values from the last traversal at the root: [2, num_hands].
  Pair<std::vector<double>> root_values;
  Pair<std::vector<double>> root_values_means;

  // Buffer to store reach probabilties for the last_strategies.
  std::vector<std::vector<double>> reach_probabilities_buffer;
};
}  // namespace

TreeStrategy get_uniform_strategy(const Game& game, const Tree& tree) {
  TreeStrategy strategy;
  init_nd(tree.size(), game.num_hands(), game.num_actions(), 0.0, &strategy);
  for (size_t node_id = 0; node_id < tree.size(); ++node_id) {
    int first = game.get_bid_range(tree[node_id].state).first;
    int last = first + tree[node_id].num_children();
    for (int hand = 0; hand < game.num_hands(); ++hand) {
      std::fill(strategy[node_id][hand].begin() + first,
                strategy[node_id][hand].begin() + last, 1. / (last - first));
    }
  }
  return strategy;
}

void print_strategy(const Game& game, const Tree& tree,
                    const TreeStrategy& strategy, std::ostream& stream) {
  assert(tree.size() == strategy.size());
  stream << "Printing strategies per node\n";
  stream.setf(std::ios_base::fixed, std::ios_base::floatfield);
  const auto old_precision = stream.precision(2);
  for (size_t node_id = 0; node_id < strategy.size(); ++node_id) {
    auto state = tree[node_id].state;
    if (!tree[node_id].num_children()) continue;
    stream << "Node=" << node_id << "\t" << game.state_to_string(state);

    for (size_t hand = 0; hand < strategy[node_id].size(); ++hand) {
      stream << "| hand=" << hand << " ";
      for (auto val : strategy[node_id][hand]) {
        stream << val << " ";
      }
    }
    stream << "\n";
  }
  stream.precision(old_precision);
}

void print_strategy(const Game& game, const Tree& tree,
                    const TreeStrategy& strategy) {
  return print_strategy(game, tree, strategy, std::cout);
}

void print_strategy(const Game& game, const Tree& tree,
                    const TreeStrategy& strategy, const std::string& path) {
  std::ofstream f(path);
  return print_strategy(game, tree, strategy, f);
}

// POKER VARIANT: Compute win probability for poker hands
// This replaces the dice matching logic with poker hand evaluation
std::vector<double> compute_win_probability(
    const Game& game, const PartialPublicState& state,
    const std::vector<double>& beliefs) {
  // Extract board cards from state
  std::vector<int> board_cards;
  for (int i = 0; i < state.num_board_cards; ++i) {
    if (state.board_cards[i] >= 0) {  // Valid card
      board_cards.push_back(state.board_cards[i]);
    }
  }
  
  // For each possible hand, compute win probability
  std::vector<double> win_probs(game.num_hands(), 0.0);
  
  // If no board cards yet (shouldn't happen at terminal, but handle gracefully)
  if (board_cards.size() < 4) {
    // Not enough board cards - return uniform probabilities
    for (int hand = 0; hand < static_cast<int>(beliefs.size()); ++hand) {
      win_probs[hand] = 0.5;
    }
    return win_probs;
  }
  
  // For each possible my hand (pre-discard 3-card combination):
  for (int my_hand = 0; my_hand < game.num_hands(); ++my_hand) {
    // Try all discard choices and take best (player chooses optimal discard)
    int64_t my_best_hand = 0;
    for (int discard = 0; discard < 3; ++discard) {
      std::vector<int> my_hole = Game::get_post_discard_cards(my_hand, discard);
      int64_t hand_value = Game::evaluate_best_hand(my_hole, board_cards);
      if (Game::compare_hands(hand_value, my_best_hand) > 0) {
        my_best_hand = hand_value;
      }
    }
    
    // Compare against all opponent hands weighted by beliefs
    double win_prob = 0.0;
    double tie_prob = 0.0;
    double total_belief = 0.0;
    
    for (int op_hand = 0; op_hand < game.num_hands(); ++op_hand) {
      if (beliefs[op_hand] < 1e-10) continue;  // Skip zero probability
      total_belief += beliefs[op_hand];
      
      // Try all discard choices for opponent (opponent also chooses optimal discard)
      int64_t op_best_hand = 0;
      for (int discard = 0; discard < 3; ++discard) {
        std::vector<int> op_hole = Game::get_post_discard_cards(op_hand, discard);
        int64_t hand_value = Game::evaluate_best_hand(op_hole, board_cards);
        if (Game::compare_hands(hand_value, op_best_hand) > 0) {
          op_best_hand = hand_value;
        }
      }
      
      int cmp = Game::compare_hands(my_best_hand, op_best_hand);
      if (cmp > 0) {
        win_prob += beliefs[op_hand];
      } else if (cmp == 0) {
        tie_prob += beliefs[op_hand];
      }
    }
    
    // Win probability = P(win) + 0.5 * P(tie)
    if (total_belief > 1e-10) {
      win_probs[my_hand] = win_prob / total_belief + 0.5 * tie_prob / total_belief;
    } else {
      win_probs[my_hand] = 0.5;
    }
  }
  
  return win_probs;
}

std::unique_ptr<ISubgameSolver> build_solver(
    const Game& game, const PartialPublicState& root,
    const Pair<std::vector<double>>& beliefs,
    const SubgameSolvingParams& params, std::shared_ptr<IValueNet> net) {
  if (params.use_cfr) {
    return std::make_unique<CFR>(game, root, net, beliefs, params);
  } else {
    return std::make_unique<FP>(game, root, net, beliefs, params);
  }
}

std::array<double, 2> compute_exploitability2(const Game& game,
                                              const TreeStrategy& strategy) {
  const auto root = game.get_initial_state();
  const auto tree = unroll_tree(game, root, /*max_depth=*/1000000);
  Pair<std::vector<double>> beliefs;
  for (auto i : {0, 1}) {
    beliefs[i].assign(game.num_hands(), 1. / game.num_hands());
  }
  BRSolver solver(game, tree, /*value_net=*/nullptr);
  std::vector<double> values0, values1;
  solver.compute_br(/*traverser=*/0, strategy, beliefs, &values0);
  solver.compute_br(/*traverser=*/1, strategy, beliefs, &values1);
  return {vector_sum(values0) / values0.size(),
          vector_sum(values1) / values1.size()};
}

double compute_exploitability(const Game& game, const TreeStrategy& strategy) {
  auto exploitabilites = compute_exploitability2(game, strategy);
  return (exploitabilites[0] + exploitabilites[1]) / 2.0;
}

TreeStrategyStats compute_stategy_stats(const Game& game,
                                        const TreeStrategy& strategy) {
  const auto uniform_beliefs = get_initial_beliefs(game).at(0);
  const auto tree = unroll_tree(game);
  TreeStrategyStats stats;
  stats.tree = tree;

  auto& reach_probabilities = stats.reach_probabilities;
  init_nd(tree.size(), game.num_hands(), 0.0, &reach_probabilities[0]);
  init_nd(tree.size(), game.num_hands(), 0.0, &reach_probabilities[1]);
  auto& tree_values = stats.values;
  init_nd(tree.size(), game.num_hands(), 0.0, &tree_values[0]);
  init_nd(tree.size(), game.num_hands(), 0.0, &tree_values[1]);
  stats.node_reach.resize(tree.size());
  stats.node_values[0].resize(tree.size());
  stats.node_values[1].resize(tree.size());
  for (int player : {0, 1}) {
    compute_reach_probabilities(tree, strategy, uniform_beliefs, player,
                                &reach_probabilities[player]);
  }
  for (size_t node_id = tree.size(); node_id-- > 0;) {
    stats.node_reach[node_id] = vector_sum(reach_probabilities[0][node_id]) *
                                vector_sum(reach_probabilities[1][node_id]);
  }
  for (int player : {0, 1}) {
    for (size_t node_id = tree.size(); node_id-- > 0;) {
      const auto& node = tree[node_id];
      const auto& state = node.state;
      std::vector<double>& node_values = tree_values[player][node_id];
      const auto op_reach_probabilities =
          reach_probabilities[1 - player][node_id];
      std::vector<double> op_beliefs = normalize_probabilities_safe(
          op_reach_probabilities, kReachSmoothingEps);
      // POKER VARIANT: Pass state instead of last_bid
      if (game.is_terminal(state)) {
        node_values = compute_expected_terminal_values(
            game, state, /*inverse=*/Game::get_active_player(state) != player, op_beliefs);
      } else {
        assert(node.num_children() > 0);
      }
      if (Game::get_active_player(state) == player) {
        for (int hand = 0; hand < game.num_hands(); ++hand) {
          for (auto [child_node_id, action] : ChildrenActionIt(node, game)) {
            tree_values[player][node_id][hand] +=
                strategy[node_id][hand][action] *
                tree_values[player][child_node_id][hand];
          }
        }
      } else {
        for (auto [child_node_id, action] : ChildrenActionIt(node, game)) {
          double action_prob = 0;
          // Iterating over op's hands.
          for (int hand = 0; hand < game.num_hands(); ++hand) {
            action_prob += strategy[node_id][hand][action] * op_beliefs[hand];
          }
          // Iterating over traverser's hands.
          for (int hand = 0; hand < game.num_hands(); ++hand) {
            tree_values[player][node_id][hand] +=
                action_prob * tree_values[player][child_node_id][hand];
          }
        }
      }
    }
  }
  for (int player : {0, 1}) {
    for (size_t node_id = tree.size(); node_id-- > 0;) {
      auto beliefs = normalize_probabilities_safe(
          reach_probabilities[player][node_id], 1e-6);
      for (int hand = 0; hand < game.num_hands(); ++hand) {
        stats.node_values[player][node_id] +=
            beliefs[hand] * tree_values[player][node_id][hand];
      }
    }
  }

  return stats;
}

std::vector<float> get_query(const Game& game, int traverser,
                             const PartialPublicState& state,
                             const std::vector<double>& reaches1,
                             const std::vector<double>& reaches2) {
  std::vector<float> query(get_query_size(game));
  write_query_to(game, traverser, state, reaches1, reaches2, query.data());
  return query;
}

// POKER VARIANT: Updated to deserialize board cards and discard info
std::tuple<int, PartialPublicState, std::vector<double>, std::vector<double>>
deserialize_query(const Game& game, const float* query) {
  int index = 0;
  PartialPublicState state;
  // SOLVER: Deserialize active player
  state.player_id = static_cast<int>(query[index++] + 0.5);
  const int traverser = query[index++] + 0.5;
  
  // Deserialize last action (one-hot)
  // Note: kInitialAction is -1, defined in poker.h
  state.last_action = -1;  // kInitialAction
  for (Action action = 0; action < game.num_actions(); ++action) {
    if (query[index++] > 0.5) {
      state.last_action = action;
    }
  }
  
  // POKER: Deserialize board cards
  state.num_board_cards = 0;
  for (int i = 0; i < 6; ++i) {
    int card = static_cast<int>(query[index++] + 0.5);
    if (card >= 0 && card < 52) {
      state.board_cards[i] = card;
      state.num_board_cards = i + 1;
    } else {
      state.board_cards[i] = -1;
    }
  }
  
  // POKER: Deserialize discard choices
  state.discard_choice[0] = static_cast<int>(query[index++] + 0.5);
  state.discard_choice[1] = static_cast<int>(query[index++] + 0.5);
  
  // SOLVER: Deserialize street (removed pips/stacks/button)
  state.street = static_cast<int>(query[index++] + 0.5);
  
  // Deserialize beliefs
  std::vector<std::vector<double>> beliefs(2);
  for (int i = 0; i < game.num_hands(); ++i)
    beliefs[0].push_back(query[index++]);
  for (int i = 0; i < game.num_hands(); ++i)
    beliefs[1].push_back(query[index++]);
  return std::make_tuple(traverser, state, beliefs[0], beliefs[1]);
}

std::vector<double> compute_ev(const Game& game, const TreeStrategy& strategy1,
                               const TreeStrategy& strategy2) {
  auto tree = unroll_tree(game);
  assert(tree.size() == strategy1.size());
  assert(tree.size() == strategy2.size());
  std::vector<std::vector<double>> op_reach_probabilities;
  init_nd(tree.size(), game.num_hands(), 0.0, &op_reach_probabilities);
  // values[node][hand] :=
  // sum_{z, node->z} sum_{op_hand}
  //  P(op_hand) pi^{-i}(z|op_hand) pi^{i}(node -> z|hand) U_i(hand, op_hand, z)
  std::vector<std::vector<double>> values(tree.size());
  const int player = 0;
  compute_reach_probabilities(tree, strategy2, get_initial_beliefs(game)[0],
                              1 - player, &op_reach_probabilities);

  for (size_t node_id = tree.size(); node_id-- > 0;) {
    const auto& node = tree[node_id];
    const auto& state = node.state;
    if (node.num_children() == 0) {
      // POKER VARIANT: Pass state instead of last_bid
      assert(game.is_terminal(state));
      values[node_id] = compute_expected_terminal_values(
          game, state, /*inverse=*/Game::get_active_player(state) != player,
          op_reach_probabilities[node_id]);
    } else if (Game::get_active_player(state) == player) {
      values[node_id].resize(game.num_hands());
      for (auto [child_node_id, action] : ChildrenActionIt(node, game)) {
        for (int hand = 0; hand < game.num_hands(); ++hand) {
          values[node_id][hand] +=
              strategy1[node_id][hand][action] * values[child_node_id][hand];
        }
      }
    } else {
      values[node_id].resize(game.num_hands());
      for (auto child_node_id : ChildrenIt(node)) {
        for (int hand = 0; hand < game.num_hands(); ++hand) {
          values[node_id][hand] += values[child_node_id][hand];
        }
      }
    }
  }
  return values[0];
}

Pair<double> compute_ev2(const Game& game, const TreeStrategy& strategy1,
                         const TreeStrategy& strategy2) {
  auto ev1 =
      vector_sum(compute_ev(game, strategy1, strategy2)) / game.num_hands();
  auto ev2 =
      -vector_sum(compute_ev(game, strategy2, strategy1)) / game.num_hands();
  return std::array<double, 2>{ev1, ev2};
}

std::vector<std::vector<double>> compute_immediate_regrets(
    const Game& game, const std::vector<TreeStrategy>& strategies) {
  const Tree tree = unroll_tree(game);
  assert(!strategies.empty());
  TreeStrategy regrets;
  init_nd(tree.size(), game.num_hands(), game.num_actions(), 0.0, &regrets);
  PartialTreeTraverser tree_traverser(game, tree, nullptr);
  const std::vector<double> initial_beliefs = get_initial_beliefs(game)[0];
  for (size_t strategy_id = 0; strategy_id < strategies.size(); ++strategy_id) {
    const auto& last_strategies = strategies[strategy_id];
    tree_traverser.precompute_reaches(last_strategies, initial_beliefs, 0);
    tree_traverser.precompute_reaches(last_strategies, initial_beliefs, 1);
    for (int traverser : {0, 1}) {
      tree_traverser.precompute_all_leaf_values(traverser);
      for (size_t public_node = tree.size(); public_node-- > 0;) {
        const auto& node = tree[public_node];
        if (!node.num_children()) {
          // All leaf values are set by precompute_all_leaf_values.
          continue;
        }
        const auto& state = node.state;
        auto& value = tree_traverser.traverser_values[public_node];
        value.assign(value.size(), 0.0);
        if (Game::get_active_player(state) == traverser) {
          for (auto [child_node, action] : ChildrenActionIt(node, game)) {
            const auto& action_value =
                tree_traverser.traverser_values[child_node];
            for (int hand = 0; hand < game.num_hands(); ++hand) {
              regrets[public_node][hand][action] += action_value[hand];
              value[hand] += action_value[hand] *
                             last_strategies[public_node][hand][action];
            }
          }
          for (int hand = 0; hand < game.num_hands(); ++hand) {
            for (auto [child_node, action] : ChildrenActionIt(node, game)) {
              regrets[public_node][hand][action] -= value[hand];
            }
          }
        } else {
          assert(Game::get_active_player(state) == 1 - traverser);
          for (auto child_node : ChildrenIt(node)) {
            const auto& action_value =
                tree_traverser.traverser_values[child_node];
            for (int hand = 0; hand < game.num_hands(); ++hand) {
              value[hand] += action_value[hand];
            }
          }
        }
      }
    }
  }
  std::vector<std::vector<double>> immediate_regrets;
  init_nd(tree.size(), game.num_hands(), 0.0, &immediate_regrets);
  for (size_t public_node = tree.size(); public_node-- > 0;) {
    const auto& node = tree[public_node];
    if (!node.num_children()) {
      continue;
    }
    for (int hand = 0; hand < game.num_hands(); ++hand) {
      immediate_regrets[public_node][hand] =
          *std::max_element(regrets[public_node][hand].begin(),
                            regrets[public_node][hand].end()) /
          strategies.size();
    }
  }
  return immediate_regrets;
}

}  // namespace poker
