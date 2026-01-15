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

// POKER VARIANT: Test file for Toss or Hold'em poker
#include <gtest/gtest.h>

#include "poker.h"

using namespace poker;

class GameTest : public ::testing::Test {
 protected:
  // POKER: Game() takes no parameters
  const Game game;
  const PartialPublicState root;

  GameTest() : game(), root(game.get_initial_state()) {}
};

// POKER: Test action unpacking for poker actions
TEST_F(GameTest, TestUnpackAction) {
  {
    // Test FOLD action
    auto unpacked = game.unpack_action(Game::kActionFold);
    ASSERT_EQ(unpacked.type, 0);  // FOLD
    ASSERT_EQ(unpacked.amount, 0);
  }
  {
    // Test CALL/CHECK action
    auto unpacked = game.unpack_action(Game::kActionCallCheck);
    ASSERT_EQ(unpacked.type, 1);  // CALL/CHECK
    ASSERT_EQ(unpacked.amount, 0);
  }
  {
    // Test BET action (first bet size)
    auto unpacked = game.unpack_action(Game::kActionBetBase);
    ASSERT_EQ(unpacked.type, 2);  // BET/RAISE
    ASSERT_EQ(unpacked.amount, Game::kBetSizes[0]);
  }
  {
    // Test DISCARD action (discard card 0)
    auto unpacked = game.unpack_action(Game::kActionDiscardBase);
    ASSERT_EQ(unpacked.type, 3);  // DISCARD
    ASSERT_EQ(unpacked.amount, 0);
  }
}

// POKER: Test initial state
TEST_F(GameTest, TestRoot) {
  // SOLVER: Check initial state
  ASSERT_EQ(root.player_id, 0);  // Player 0 acts first
  ASSERT_EQ(root.street, 0);  // PREFLOP
  ASSERT_EQ(root.num_board_cards, 0);
  ASSERT_EQ(root.discard_complete, false);
  
  {
    // Pre-flop: all betting actions available
    auto range = game.get_bid_range(root);
    ASSERT_EQ(range.first, 0);
    ASSERT_EQ(range.second, Game::kActionDiscardBase);  // All betting actions
  }
}

// POKER: Test state transitions
TEST_F(GameTest, TestStateTransitions) {
  auto state = root;
  
  // Pre-flop action: call/check
  state = game.act(state, Game::kActionCallCheck);
  ASSERT_EQ(state.player_id, 1);  // Now player 1 acts
  
  // After discard round, should move to flop betting
  // (This is simplified - full implementation needs proper state tracking)
}

// POKER: Test poker hand evaluation
TEST_F(GameTest, TestPokerHandEvaluation) {
  // Test high card
  std::vector<int> cards = {0, 5, 10, 15, 20};  // 2c, 3d, 4h, 5s, 6c
  int64_t hand_value = Game::evaluate_5card_hand(cards);
  int hand_type = Game::get_hand_type(hand_value);
  ASSERT_EQ(hand_type, 0);  // High card
  
  // Test pair
  cards = {0, 4, 10, 15, 20};  // 2c, 2d, 4h, 5s, 6c
  hand_value = Game::evaluate_5card_hand(cards);
  hand_type = Game::get_hand_type(hand_value);
  ASSERT_EQ(hand_type, 1);  // Pair
  
  // Test two pair
  cards = {0, 4, 8, 12, 20};  // 2c, 2d, 3c, 3d, 6c
  hand_value = Game::evaluate_5card_hand(cards);
  hand_type = Game::get_hand_type(hand_value);
  ASSERT_EQ(hand_type, 2);  // Two pair
  
  // Test three of a kind
  cards = {0, 4, 8, 15, 20};  // 2c, 2d, 2h, 5s, 6c
  hand_value = Game::evaluate_5card_hand(cards);
  hand_type = Game::get_hand_type(hand_value);
  ASSERT_EQ(hand_type, 3);  // Three of a kind
}

// POKER: Test hand to cards conversion
TEST_F(GameTest, TestHandToCards) {
  // Test first hand (should be cards 0, 1, 2)
  auto cards = Game::hand_to_cards(0);
  ASSERT_EQ(cards.size(), 3);
  // Should be the first 3 cards in lexicographic order
  ASSERT_EQ(cards[0], 0);
  ASSERT_EQ(cards[1], 1);
  ASSERT_EQ(cards[2], 2);
  
  // Test a later hand
  auto cards2 = Game::hand_to_cards(100);
  ASSERT_EQ(cards2.size(), 3);
  // All cards should be valid (0-51)
  for (int card : cards2) {
    ASSERT_GE(card, 0);
    ASSERT_LT(card, 52);
  }
}

// POKER: Test post-discard cards
TEST_F(GameTest, TestPostDiscardCards) {
  int hand = 0;  // First hand: cards 0, 1, 2
  auto post_discard = Game::get_post_discard_cards(hand, 0);  // Discard card 0
  ASSERT_EQ(post_discard.size(), 2);
  ASSERT_EQ(post_discard[0], 1);  // Card 1
  ASSERT_EQ(post_discard[1], 2);  // Card 2
}

// POKER: Test best hand evaluation
TEST_F(GameTest, TestBestHand) {
  std::vector<int> hole = {0, 4};  // 2c, 2d
  std::vector<int> board = {8, 12, 16, 20, 24};  // 3c, 3d, 4c, 5c, 6c
  
  int64_t best = Game::evaluate_best_hand(hole, board);
  int hand_type = Game::get_hand_type(best);
  ASSERT_EQ(hand_type, 2);  // Two pair (2s and 3s)
}
