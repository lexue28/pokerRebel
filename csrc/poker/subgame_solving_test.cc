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

// POKER VARIANT: Updated tests for poker subgame solving
#include <math.h>

#include <gtest/gtest.h>

#include "real_net.h"
#include "subgame_solving.h"
#include "util.h"

using namespace poker;

namespace {
double compute_fp_exploitability(const Game& game,
                                 const PartialPublicState& root,
                                 const Pair<std::vector<double>>& beliefs,
                                 const SubgameSolvingParams& params,
                                 std::shared_ptr<IValueNet> net) {
  assert(beliefs[0].size() == static_cast<size_t>(game.num_hands()));
  assert(beliefs[1].size() == static_cast<size_t>(game.num_hands()));
  auto solver = build_solver(game, root, beliefs, params, net);
  solver->multistep();
  std::array<double, 2> values =
      compute_exploitability2(game, solver->get_strategy());
  return (values[0] + values[1]) / 2.;
}
// No-network version assumes that all leaf nodes are final.
double compute_fp_exploitability(const Game& game,
                                 const PartialPublicState& root,
                                 const Pair<std::vector<double>>& beliefs,
                                 const SubgameSolvingParams& params) {
  return compute_fp_exploitability(game, root, beliefs, params, nullptr);
}
}  // namespace

// POKER: Test poker hand evaluation
TEST(PokerHandEvaluation, TestHighCard) {
  std::vector<int> cards = {0, 5, 10, 15, 20};  // 2c, 3d, 4h, 5s, 6c
  int64_t hand_value = Game::evaluate_5card_hand(cards);
  int hand_type = Game::get_hand_type(hand_value);
  ASSERT_EQ(hand_type, 0);  // High card
}

TEST(PokerHandEvaluation, TestPair) {
  std::vector<int> cards = {0, 4, 10, 15, 20};  // 2c, 2d, 4h, 5s, 6c
  int64_t hand_value = Game::evaluate_5card_hand(cards);
  int hand_type = Game::get_hand_type(hand_value);
  ASSERT_EQ(hand_type, 1);  // Pair
}

TEST(PokerHandEvaluation, TestTwoPair) {
  std::vector<int> cards = {0, 4, 8, 12, 20};  // 2c, 2d, 3c, 3d, 6c
  int64_t hand_value = Game::evaluate_5card_hand(cards);
  int hand_type = Game::get_hand_type(hand_value);
  ASSERT_EQ(hand_type, 2);  // Two pair
}

TEST(PokerHandEvaluation, TestThreeOfAKind) {
  std::vector<int> cards = {0, 4, 8, 15, 20};  // 2c, 2d, 2h, 5s, 6c
  int64_t hand_value = Game::evaluate_5card_hand(cards);
  int hand_type = Game::get_hand_type(hand_value);
  ASSERT_EQ(hand_type, 3);  // Three of a kind
}

TEST(PokerHandEvaluation, TestStraight) {
  std::vector<int> cards = {0, 5, 10, 15, 20};  // 2c, 3d, 4h, 5s, 6c
  int64_t hand_value = Game::evaluate_5card_hand(cards);
  int hand_type = Game::get_hand_type(hand_value);
  ASSERT_EQ(hand_type, 4);  // Straight
}

TEST(PokerHandEvaluation, TestFlush) {
  std::vector<int> cards = {0, 4, 8, 12, 16};  // All clubs: 2c, 3c, 4c, 5c, 6c
  int64_t hand_value = Game::evaluate_5card_hand(cards);
  int hand_type = Game::get_hand_type(hand_value);
  ASSERT_EQ(hand_type, 5);  // Flush
}

TEST(PokerHandEvaluation, TestFullHouse) {
  std::vector<int> cards = {0, 4, 8, 12, 16};  // 2c, 2d, 2h, 3c, 3d
  // Actually need 3 of one rank and 2 of another
  cards = {0, 4, 8, 12, 13};  // 2c, 2d, 2h, 3c, 3d
  int64_t hand_value = Game::evaluate_5card_hand(cards);
  int hand_type = Game::get_hand_type(hand_value);
  ASSERT_EQ(hand_type, 6);  // Full house
}

TEST(PokerHandEvaluation, TestHandComparison) {
  std::vector<int> pair = {0, 4, 10, 15, 20};  // Pair of 2s
  std::vector<int> high_card = {1, 5, 10, 15, 20};  // High card
  int64_t pair_value = Game::evaluate_5card_hand(pair);
  int64_t high_card_value = Game::evaluate_5card_hand(high_card);
  ASSERT_GT(pair_value, high_card_value);  // Pair beats high card
}

// POKER: Test win probability computation with poker hands
TEST(PokerWinProbability, TestSimpleShowdown) {
  Game game;
  
  // Create a terminal state with board cards
  PartialPublicState state = game.get_initial_state();
  state.street = 999;  // Terminal (showdown)
  state.num_board_cards = 6;
  // Set up board cards (simplified - in real game these come from game flow)
  for (int i = 0; i < 6; ++i) {
    state.board_cards[i] = 10 + i;  // Some board cards
  }
  
  // Test with uniform beliefs
  std::vector<double> beliefs(game.num_hands(), 1.0 / game.num_hands());
  auto values = compute_win_probability(game, state, beliefs);
  
  ASSERT_EQ(values.size(), game.num_hands());
  // All values should be probabilities (0-1)
  for (double v : values) {
    ASSERT_GE(v, 0.0);
    ASSERT_LE(v, 1.0);
  }
}

// POKER: Test FP solving on small subgame
TEST(FictiousTest, TestSmallSubgame) {
  // POKER: Game() takes no parameters
  SubgameSolvingParams params;
  params.num_iters = 100;
  params.max_depth = 2;  // Small subgame
  params.linear_update = true;
  Game game;

  const auto root = game.get_initial_state();
  const auto initial_beliefs = get_initial_beliefs(game);
  
  const auto value =
      compute_fp_exploitability(game, root, initial_beliefs, params);
  ASSERT_GE(value, 0.0);
  // POKER: Exploitability should be reasonable (not too high)
  ASSERT_LT(value, 1.0);
}

TEST(FictiousTest, TestSmallSubgameLinear) {
  // POKER: Test with linear update
  SubgameSolvingParams params;
  params.num_iters = 100;
  params.max_depth = 2;
  params.linear_update = true;
  Game game;

  const auto root = game.get_initial_state();
  const auto initial_beliefs = get_initial_beliefs(game);
  
  const auto value =
      compute_fp_exploitability(game, root, initial_beliefs, params);
  ASSERT_GE(value, 0.0);
  ASSERT_LT(value, 1.0);
}

TEST(CFRTest, TestSmallSubgameCfr) {
  // POKER: Test CFR solving
  SubgameSolvingParams params;
  params.num_iters = 100;
  params.max_depth = 2;
  params.linear_update = true;
  params.use_cfr = true;
  Game game;

  const auto root = game.get_initial_state();
  const auto initial_beliefs = get_initial_beliefs(game);
  
  const auto value =
      compute_fp_exploitability(game, root, initial_beliefs, params);
  ASSERT_GE(value, 0.0);
  ASSERT_LT(value, 1.0);
}

TEST(QueryTest, TestQueryDeserialization) {
  // POKER: Test query serialization/deserialization
  Game game;
  const auto tree = unroll_tree(game, game.get_initial_state(), 3);
  std::vector<double> beliefs1, beliefs2;
  for (int i = 0; i < game.num_hands(); ++i) beliefs1.push_back(i);
  for (int i = 0; i < game.num_hands(); ++i) beliefs2.push_back(i + 0.5);
  normalize_probabilities(beliefs1, &beliefs1);
  normalize_probabilities(beliefs2, &beliefs2);
  
  for (int traverser : {0, 1}) {
    for (const auto& node : tree) {
      const auto& state = node.state;
      // Value net cannot be queried in terminal nodes.
      if (game.is_terminal(state)) continue;
      const auto query = get_query(game, traverser, state, beliefs1, beliefs2);

      const auto [deserialized_traverser, deserialized_state,
                  deserialized_beliefs1, deserialized_beliefs2] =
          deserialize_query(game, query.data());
      // SOLVER: Check active player and street
      ASSERT_EQ(state.player_id, deserialized_state.player_id);
      ASSERT_EQ(traverser, deserialized_traverser);
      ASSERT_EQ(state.last_action, deserialized_state.last_action);
      ASSERT_EQ(state.street, deserialized_state.street);
      ASSERT_EQ(state.num_board_cards, deserialized_state.num_board_cards);
      for (int i = 0; i < game.num_hands(); ++i) {
        ASSERT_NEAR(beliefs1[i], deserialized_beliefs1[i], 1e-6);
        ASSERT_NEAR(beliefs2[i], deserialized_beliefs2[i], 1e-6);
      }
    }
  }
}

TEST(UtilsTest, TestProbNormalization) {
  std::vector<double> probs{2.93185e-81, 3.00956e-81, 3.17805e-81, 8.80785e-81};
  std::vector<double> out(probs.size());
  normalize_probabilities_safe(probs, kReachSmoothingEps, out.data());
  ASSERT_NEAR(vector_sum(out), 1.0, 1e-10);
}

TEST(UtilsTest, TestProbNormalizationFloat) {
  std::vector<double> probs{2.93185e-81, 3.00956e-81, 3.17805e-81, 8.80785e-81};
  std::vector<float> out(probs.size());
  normalize_probabilities_safe(probs, kReachSmoothingEps, out.data());
  ASSERT_NEAR(vector_sum(out), 1.0, 1e-10);
}
