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

// POKER VARIANT: Adapted for Toss or Hold'em
// String formatting functions for poker actions and states

#include "poker.h"
#include "real_net.h"

#include <sstream>

namespace poker {

// Convert action to human-readable string
std::string Game::action_to_string(Action action) const {
  UnpackedAction unpacked = unpack_action(action);
  
  if (unpacked.type == 0) {  // FOLD
    return "fold";
  } else if (unpacked.type == 1) {  // CALL/CHECK
    return "call/check";
  } else if (unpacked.type == 2) {  // BET/RAISE
    std::ostringstream ss;
    ss << "bet/raise(" << unpacked.amount << ")";
    return ss.str();
  } else if (unpacked.type == 3) {  // DISCARD
    std::ostringstream ss;
    ss << "discard(card=" << unpacked.amount << ")";
    return ss.str();
  }
  return "unknown";
}

// Convert state to human-readable string
std::string Game::state_to_string(const PartialPublicState& state) const {
  std::ostringstream ss;
  const std::string last_action_str = state.last_action == kInitialAction
                                          ? "start"
                                          : action_to_string(state.last_action);
  ss << "(street=" << state.street 
     << ",player=" << state.player_id
     << ",board=" << state.num_board_cards
     << ",last=" << last_action_str << ")";
  return ss.str();
}

// Convert action to short string
std::string Game::action_to_string_short(Action action) const {
  UnpackedAction unpacked = unpack_action(action);
  
  if (unpacked.type == 0) {  // FOLD
    return "F";
  } else if (unpacked.type == 1) {  // CALL/CHECK
    return "C";
  } else if (unpacked.type == 2) {  // BET/RAISE
    std::ostringstream ss;
    ss << "B" << unpacked.amount;
    return ss.str();
  } else if (unpacked.type == 3) {  // DISCARD
    std::ostringstream ss;
    ss << "D" << unpacked.amount;
    return ss.str();
  }
  return "?";
}

// Convert state to short string
std::string Game::state_to_string_short(const PartialPublicState& state) const {
  std::ostringstream ss;
  const std::string last_action_str = state.last_action == kInitialAction
                                          ? "beg"
                                          : action_to_string_short(state.last_action);
  ss << "S" << state.street
     << "p" << state.player_id
     << "," << last_action_str;
  return ss.str();
}

// POKER HAND EVALUATION IMPLEMENTATIONS

namespace {
// Helper: Get all combinations of k elements from n elements
void get_combinations(int n, int k, int start, std::vector<int>& current,
                     std::vector<std::vector<int>>& result) {
  if (current.size() == static_cast<size_t>(k)) {
    result.push_back(current);
    return;
  }
  for (int i = start; i < n; ++i) {
    current.push_back(i);
    get_combinations(n, k, i + 1, current, result);
    current.pop_back();
  }
}
}  // namespace

// Convert hand index to actual card indices
// Uses combination indexing: C(52, 3) combinations
std::vector<int> Game::hand_to_cards(int hand) {
  // hand is an index into all C(52, 3) = 22,100 combinations
  // We need to find which 3 cards this represents
  std::vector<int> cards;
  int remaining = hand;
  int prev_card = -1;
  
  for (int i = 0; i < 3; ++i) {
    int card = prev_card + 1;
    while (true) {
      // Number of combinations remaining if we choose this card
      int cards_left = 3 - i - 1;
      int cards_available = 52 - card - 1;
      if (cards_left == 0) {
        // Last card, just use remaining
        cards.push_back(card);
        break;
      }
      // C(cards_available, cards_left)
      int64_t combos = 1;
      for (int j = 0; j < cards_left; ++j) {
        combos = combos * (cards_available - j) / (j + 1);
      }
      if (remaining < combos) {
        cards.push_back(card);
        prev_card = card;
        break;
      }
      remaining -= combos;
      ++card;
    }
  }
  return cards;
}

// Get post-discard hole cards
std::vector<int> Game::get_post_discard_cards(int pre_discard_hand, int discard_index) {
  std::vector<int> pre_discard = hand_to_cards(pre_discard_hand);
  std::vector<int> post_discard;
  for (int i = 0; i < 3; ++i) {
    if (i != discard_index) {
      post_discard.push_back(pre_discard[i]);
    }
  }
  return post_discard;
}

// Evaluate 5-card poker hand
// Returns encoded hand rank: (hand_type << 20) | (kicker bits)
int64_t Game::evaluate_5card_hand(const std::vector<int>& cards) {
  assert(cards.size() == 5);
  
  // Count ranks and suits
  int rank_count[13] = {0};
  int suit_count[4] = {0};
  for (int card : cards) {
    int rank = card_rank(card);
    int suit = card_suit(card);
    rank_count[rank]++;
    suit_count[suit]++;
  }
  
  // Check for flush
  bool is_flush = false;
  for (int s = 0; s < 4; ++s) {
    if (suit_count[s] == 5) {
      is_flush = true;
      break;
    }
  }
  
  // Check for straight
  bool is_straight = false;
  int straight_high = -1;
  // Check A-2-3-4-5 (wheel)
  if (rank_count[0] && rank_count[1] && rank_count[2] && rank_count[3] && rank_count[12]) {
    is_straight = true;
    straight_high = 3;  // 5-high straight
  }
  // Check regular straights
  for (int start = 0; start <= 8; ++start) {
    bool all_present = true;
    for (int i = 0; i < 5; ++i) {
      if (rank_count[start + i] == 0) {
        all_present = false;
        break;
      }
    }
    if (all_present) {
      is_straight = true;
      straight_high = start + 4;
      break;
    }
  }
  
  // Count pairs, trips, quads
  int pairs = 0, trips = 0, quads = 0;
  int pair_ranks[2] = {-1, -1};
  int trip_rank = -1;
  int quad_rank = -1;
  int kickers[5];
  int kicker_idx = 0;
  
  for (int r = 12; r >= 0; --r) {
    if (rank_count[r] == 4) {
      quads++;
      quad_rank = r;
    } else if (rank_count[r] == 3) {
      trips++;
      trip_rank = r;
    } else if (rank_count[r] == 2) {
      if (pairs < 2) {
        pair_ranks[pairs] = r;
      }
      pairs++;
    } else if (rank_count[r] == 1) {
      if (kicker_idx < 5) {
        kickers[kicker_idx++] = r;
      }
    }
  }
  
  // Encode hand value
  int64_t hand_value = 0;
  
  // Straight flush (including royal flush)
  if (is_straight && is_flush) {
    hand_value = (8LL << 20) | straight_high;
  }
  // Four of a kind
  else if (quads > 0) {
    hand_value = (7LL << 20) | (quad_rank << 4) | kickers[0];
  }
  // Full house
  else if (trips > 0 && pairs > 0) {
    hand_value = (6LL << 20) | (trip_rank << 4) | pair_ranks[0];
  }
  // Flush
  else if (is_flush) {
    hand_value = (5LL << 20);
    for (int i = 0; i < 5; ++i) {
      hand_value |= (kickers[i] << (4 * (4 - i)));
    }
  }
  // Straight
  else if (is_straight) {
    hand_value = (4LL << 20) | straight_high;
  }
  // Three of a kind
  else if (trips > 0) {
    hand_value = (3LL << 20) | (trip_rank << 8) | (kickers[0] << 4) | kickers[1];
  }
  // Two pair
  else if (pairs >= 2) {
    hand_value = (2LL << 20) | (pair_ranks[0] << 8) | (pair_ranks[1] << 4) | kickers[0];
  }
  // One pair
  else if (pairs == 1) {
    hand_value = (1LL << 20) | (pair_ranks[0] << 12) | (kickers[0] << 8) | (kickers[1] << 4) | kickers[2];
  }
  // High card
  else {
    hand_value = (0LL << 20) | (kickers[0] << 16) | (kickers[1] << 12) | (kickers[2] << 8) | (kickers[3] << 4) | kickers[4];
  }
  
  return hand_value;
}

// Evaluate best hand from 2 hole cards + board cards
int64_t Game::evaluate_best_hand(const std::vector<int>& hole_cards,
                                  const std::vector<int>& board_cards) {
  assert(hole_cards.size() == 2);
  
  // Collect all available cards
  std::vector<int> all_cards = hole_cards;
  for (int card : board_cards) {
    if (card >= 0) {  // Valid card
      all_cards.push_back(card);
    }
  }
  
  // Need to choose best 5 cards from available
  int64_t best_hand = 0;
  
  if (all_cards.size() == 5) {
    best_hand = evaluate_5card_hand(all_cards);
  } else if (all_cards.size() > 5) {
    // Try all combinations of 5 cards
    std::vector<std::vector<int>> combinations;
    std::vector<int> current;
    get_combinations(all_cards.size(), 5, 0, current, combinations);
    
    for (const auto& combo : combinations) {
      std::vector<int> hand_cards;
      for (int idx : combo) {
        hand_cards.push_back(all_cards[idx]);
      }
      int64_t hand_value = evaluate_5card_hand(hand_cards);
      if (compare_hands(hand_value, best_hand) > 0) {
        best_hand = hand_value;
      }
    }
  }
  
  return best_hand;
}

// Compare two hand ranks
int Game::compare_hands(int64_t hand1_rank, int64_t hand2_rank) {
  if (hand1_rank > hand2_rank) return 1;
  if (hand1_rank < hand2_rank) return -1;
  return 0;
}

// Get hand type from rank value
int Game::get_hand_type(int64_t rank) {
  return (rank >> 20) & 0xF;
}

}  // namespace poker