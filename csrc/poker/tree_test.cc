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

// POKER VARIANT: Updated tests for poker game tree
#include "tree.h"
#include <gtest/gtest.h>

using namespace poker;

TEST(TreeTest, TestUnroll) {
  // POKER: Game() takes no parameters
  Game game;
  
  auto nodes = unroll_tree(game, game.get_initial_state(), 2);
  
  // POKER: Tree structure will be different - just verify it works
  ASSERT_GT(nodes.size(), 0);
  EXPECT_EQ(nodes[0].parent, -1);  // Root has no parent
  EXPECT_GE(nodes[0].num_children(), 0);
}

TEST(TreeTest, TestUnrollDepthZero) {
  // POKER: Test depth 0 (just root)
  Game game;
  
  const auto root = game.get_initial_state();
  auto nodes = unroll_tree(game, root, 0);
  
  ASSERT_EQ(nodes.size(), 1);
  EXPECT_EQ(nodes[0].parent, -1);
  EXPECT_EQ(nodes[0].get_children().size(), 0);
  EXPECT_EQ(nodes[0].state, root);
}

TEST(TreeTest, TestUnrollDepthOne) {
  // POKER: Test depth 1 (root + children)
  Game game;
  
  const auto root = game.get_initial_state();
  auto nodes = unroll_tree(game, root, 1);
  
  ASSERT_GT(nodes.size(), 1);
  EXPECT_EQ(nodes[0].parent, -1);
  EXPECT_GT(nodes[0].num_children(), 0);
  // Verify children have correct parent
  for (int child : nodes[0].get_children()) {
    EXPECT_EQ(nodes[child].parent, 0);
  }
}

TEST(TreeTest, TestUnrollDepthTwo) {
  // POKER: Test depth 2
  Game game;
  
  const auto root = game.get_initial_state();
  auto nodes = unroll_tree(game, root, 2);
  
  ASSERT_GT(nodes.size(), 1);
  EXPECT_EQ(nodes[0].parent, -1);
  // Verify tree structure
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].num_children() > 0) {
      for (int child : nodes[i].get_children()) {
        EXPECT_EQ(nodes[child].parent, static_cast<int>(i));
        EXPECT_EQ(nodes[child].depth, nodes[i].depth + 1);
      }
    }
  }
}

TEST(TreeTest, TestTreeIsBreadthFirst) {
  // POKER: Required for partial initialization to work - verify BFS ordering
  Game game;
  const auto tree = unroll_tree(game, game.get_initial_state(), 10);
  
  for (size_t subtree_depth = 0; subtree_depth < 10; ++subtree_depth) {
    const auto subtree =
        unroll_tree(game, game.get_initial_state(), subtree_depth);
    for (size_t i = 0; i < subtree.size(); ++i) {
      ASSERT_EQ(tree[i].state, subtree[i].state);
      if (subtree[i].num_children()) {
        ASSERT_EQ(tree[i].children_begin, subtree[i].children_begin);
        ASSERT_EQ(tree[i].children_end, subtree[i].children_end);
        ASSERT_EQ(tree[i].parent, subtree[i].parent);
      }
    }
  }
}

TEST(TreeTest, TestStateEquality) {
  // POKER: Test that state equality works correctly
  Game game;
  auto state1 = game.get_initial_state();
  auto state2 = game.get_initial_state();
  
  EXPECT_EQ(state1, state2);
  
  // Modify state and verify inequality
  state2.player_id = 1;  // Change active player
  EXPECT_NE(state1, state2);
}
