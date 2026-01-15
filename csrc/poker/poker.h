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

// ============================================================================
// POKER VARIANT ADAPTATION: TOSS OR HOLD'EM
// ============================================================================
// This file has been adapted from liars_dice to implement "Toss or Hold'em"
// poker variant. Key changes:
//
// 1. Game representation: Cards (52) instead of dice with faces
// 2. Hand representation: 3 pre-discard hole cards (C(52,3) = 22,100 hands)
// 3. Actions: Betting (fold, call/check, bet/raise) + discard (choose card 0/1/2)
// 4. State: Tracks betting round, board cards, discard choices
// 5. Terminal evaluation: Poker hand comparison instead of dice matching
//
// TODO FOR PROOFREADING:
// - Verify action encoding (bet sizes, discard actions)
// - Check state transition logic in act() - currently simplified
// - Verify poker hand evaluation combination indexing
// - Check that board cards are properly tracked through state transitions
// - Verify query serialization includes all necessary information
// ============================================================================

#pragma once

#include <assert.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace poker {

using Action = int;

// POKER VARIANT: Toss or Hold'em
// This file has been adapted from liars_dice to implement Toss or Hold'em poker.
// Key changes:
// - Cards instead of dice (52 cards in standard deck)
// - Betting actions (fold, call, check, bet/raise) instead of bidding
// - Discard actions (choose which of 3 pre-discard cards to discard)
// - Poker hand evaluation instead of dice matching

// Action types for poker betting
enum class BettingActionType {
  FOLD = 0,
  CALL_OR_CHECK = 1,
  BET_OR_RAISE = 2  // Followed by bet amount
};

// Unpacked action for poker - represents betting action or discard
struct UnpackedAction {
  // For betting: type (FOLD, CALL/CHECK, or BET/RAISE) and amount
  // For discard: type is unused, amount is card index (0, 1, or 2)
  int type;  // 0=fold, 1=call/check, 2=bet/raise, 3=discard
  int amount;  // Bet amount or discard card index
};

// Betting round enumeration
enum class BettingRound {
  PREFLOP = 0,
  FLOP_DISCARD = 1,  // Discard round (not betting, but action phase)
  FLOP_BETTING = 2,
  TURN_BETTING = 3,
  RIVER_BETTING = 4,
  SHOWDOWN = 5
};

// Public state of the game for Toss or Hold'em
// NOTE: This is the SOLVER's internal state representation - separate from engine.py
// The solver only needs what's necessary for CFR/value prediction, not full game state
struct PartialPublicState {
  // Street/betting round: 0=preflop, 2=flop+discard1, 3=discard2, 4=flop_betting, 5=turn, 6=river
  // Terminal states: street = 999 (fold) or street = 6 (showdown)
  int street;
  // Player whose turn it is (0 or 1)
  int player_id;
  // Last action taken (encoded as action ID, -1 for initial state)
  Action last_action;
  // Number of board cards visible (2 after flop, 4 after discard, 5 after turn, 6 after river)
  int num_board_cards;
  // Board cards (public information) - max 6 cards
  // Cards are encoded as 0-51 (0-12 rank * 4 + 0-3 suit)
  // Unused slots are set to -1
  int board_cards[6];
  // Discard choices (which card index was discarded by each player, -1 if not discarded yet)
  // After discard, these cards become board cards
  int discard_choice[2];  // [0] = player 0's discard, [1] = player 1's discard

  bool operator==(const PartialPublicState& state) const {
    if (street != state.street ||
        player_id != state.player_id ||
        last_action != state.last_action ||
        num_board_cards != state.num_board_cards) {
      return false;
    }
    for (int i = 0; i < 2; ++i) {
      if (discard_choice[i] != state.discard_choice[i]) return false;
    }
    for (int i = 0; i < 6; ++i) {
      if (board_cards[i] != state.board_cards[i]) return false;
    }
    return true;
  }
};

class Game {
 public:
  // POKER CONSTANTS
  static constexpr int kNumCards = 52;  // Standard deck
  static constexpr int kNumPreDiscardCards = 3;  // 3 hole cards before discard
  static constexpr int kNumPostDiscardCards = 2;  // 2 hole cards after discard
  static constexpr int kStackSize = 400;  // Stack per round
  static constexpr int kSmallBlind = 1;
  static constexpr int kBigBlind = 2;
  static constexpr int kMaxBetSize = kStackSize;  // Maximum bet size (all-in)

  // For discretized betting, we'll use a set of bet sizes
  // Bet sizes as multiples of big blind: 1, 2, 4, 8, 16, 32, 64, 128, 256, 400 (all-in)
  static constexpr int kNumBetSizes = 10;
  static constexpr int kBetSizes[kNumBetSizes] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 400};

  // Action encoding:
  // Actions 0: FOLD
  // Action 1: CALL/CHECK
  // Actions 2 to 1+kNumBetSizes: BET/RAISE with bet sizes
  // Actions 2+kNumBetSizes to 4+kNumBetSizes: DISCARD card 0, 1, or 2
  static constexpr int kActionFold = 0;
  static constexpr int kActionCallCheck = 1;
  static constexpr int kActionBetBase = 2;  // Bet actions start here
  static constexpr int kActionDiscardBase = 2 + kNumBetSizes;  // Discard actions start here

  Game() 
      : num_hands_(compute_num_hands()),
        num_actions_(2 + kNumBetSizes + 3) {  // fold, call/check, bet sizes, 3 discard options
    // Constructor for poker game - no parameters needed
  }

  // Maximum number of distinct actions in every node
  Action num_actions() const { return num_actions_; }
  
  // Number of distinct game states at the beginning (number of possible 3-card hands)
  // C(52, 3) = 52! / (3! * 49!) = 22,100
  int num_hands() const { return num_hands_; }
  
  // Upper bound for how deep game tree could be
  int max_depth() const { return 1000; }  // Arbitrary large number for poker

  // Unpack a betting action
  UnpackedAction unpack_action(Action action) const {
    UnpackedAction unpacked;
    if (action == kActionFold) {
      unpacked.type = 0;  // FOLD
      unpacked.amount = 0;
    } else if (action == kActionCallCheck) {
      unpacked.type = 1;  // CALL/CHECK
      unpacked.amount = 0;
    } else if (action >= kActionBetBase && action < kActionDiscardBase) {
      unpacked.type = 2;  // BET/RAISE
      unpacked.amount = kBetSizes[action - kActionBetBase];
    } else if (action >= kActionDiscardBase && action < num_actions()) {
      unpacked.type = 3;  // DISCARD
      unpacked.amount = action - kActionDiscardBase;  // 0, 1, or 2
    } else {
      assert(false && "Invalid action");
      unpacked.type = -1;
      unpacked.amount = -1;
    }
    return unpacked;
  }

  // Get the bet size for a bet/raise action
  int get_bet_size(Action action) const {
    if (action >= kActionBetBase && action < kActionDiscardBase) {
      return kBetSizes[action - kActionBetBase];
    }
    return 0;
  }

  // Check if action is a discard action
  bool is_discard_action(Action action) const {
    return action >= kActionDiscardBase && action < num_actions();
  }

  // Get discard card index from action
  int get_discard_index(Action action) const {
    assert(is_discard_action(action));
    return action - kActionDiscardBase;
  }

  // Get initial state (after blinds posted, cards dealt)
  // SOLVER: Simplified state - no need for pips/stacks/button
  PartialPublicState get_initial_state() const {
    PartialPublicState state;
    state.street = 0;  // Pre-flop
    state.player_id = 0;  // Player 0 (dealer/small blind) acts first
    state.last_action = kInitialAction;
    state.num_board_cards = 0;
    for (int i = 0; i < 6; ++i) {
      state.board_cards[i] = -1;  // No board cards yet
    }
    state.discard_choice[0] = -1;
    state.discard_choice[1] = -1;
    return state;
  }

  // Get range of possible actions in the state as [min_action, max_action)
  // SOLVER: Simplified - just return all legal actions for the street
  std::pair<Action, Action> get_bid_range(
      const PartialPublicState& state) const {
    // During discard phase (street 2 or 3), only discard actions
    if (state.street == 2 || state.street == 3) {
      // Street 2: player 1 discards, street 3: player 0 discards
      if ((state.street == 2 && state.player_id == 1) || 
          (state.street == 3 && state.player_id == 0)) {
        return std::pair<Action, Action>(kActionDiscardBase, num_actions());
      } else {
        // Other player - no action needed (will advance street)
        return std::pair<Action, Action>(kActionCallCheck, kActionCallCheck + 1);
      }
    }
    
    // During betting rounds, return all betting actions
    // SOLVER: Don't filter by bet sizes - let the solver explore all actions
    return std::pair<Action, Action>(0, kActionDiscardBase);
  }

  // Check if state is terminal (showdown or fold)
  // Based on engine.py: terminal when street == 999 (fold) or street == 6 completes
  bool is_terminal(const PartialPublicState& state) const {
    return state.street == 999 || state.street == 6;  // Terminal marker or river complete
  }
  
  // Helper: Get active player (just return player_id)
  static int get_active_player(const PartialPublicState& state) {
    return state.player_id;
  }

  // Apply action to state and return new state
  // SOLVER: Simplified state transitions - no need to track pips/stacks/button
  PartialPublicState act(const PartialPublicState& state, Action action) const {
    const auto range = get_bid_range(state);
    assert(action >= range.first);
    assert(action < range.second);
    
    PartialPublicState new_state = state;
    UnpackedAction unpacked = unpack_action(action);
    
    // Handle DiscardAction (street 2 or 3)
    if (unpacked.type == 3) {  // DISCARD
      assert(state.street == 2 || state.street == 3);
      // Store discard choice
      new_state.discard_choice[state.player_id] = unpacked.amount;
      // Move to next player
      new_state.player_id = 1 - state.player_id;
      // If both players discarded, advance to next street
      if ((state.street == 2 && new_state.player_id == 0) ||
          (state.street == 3 && new_state.player_id == 1)) {
        return proceed_street(new_state);
      }
      return new_state;
    }
    
    // Handle FoldAction
    if (unpacked.type == 0) {  // FOLD
      // Terminal - mark as terminal
      new_state.street = 999;
      return new_state;
    }
    
    // Handle CallAction/CheckAction (type 1)
    // SOLVER: Simplified - just switch players, advance street when both acted
    if (unpacked.type == 1) {
      new_state.last_action = action;
      new_state.player_id = 1 - state.player_id;
      // If we've switched back to original player, both acted - advance street
      // (Simplified: assume betting round completes after 2 actions)
      if (new_state.player_id == 0 && state.street != 0) {
        return proceed_street(new_state);
      }
      return new_state;
    }
    
    // Handle RaiseAction/BetAction (type 2)
    // SOLVER: Simplified - just switch players
    if (unpacked.type == 2) {
      new_state.last_action = action;
      new_state.player_id = 1 - state.player_id;
      return new_state;
    }
    
    return new_state;
  }
  
  // Advance to next street
  // SOLVER: Simplified - just handle street transitions
  PartialPublicState proceed_street(PartialPublicState state) const {
    if (state.street == 6) {
      // Terminal - showdown
      state.street = 999;
      return state;
    }
    
    if (state.street == 0) {
      // Pre-flop -> Flop (street 2)
      state.street = 2;
      state.player_id = 1;  // Player 1 discards first
      state.num_board_cards = 2;  // 2 flop cards (set externally)
    } else if (state.street == 2) {
      // After player 1 discards -> Player 0 discards (street 3)
      state.street = 3;
      state.player_id = 0;  // Player 0 discards
      state.num_board_cards = 3;  // 2 flop + 1 discard
    } else if (state.street == 3) {
      // After both discards -> Flop betting (street 4)
      state.street = 4;
      state.player_id = 1;  // Player 1 acts first after discard
      state.num_board_cards = 4;  // 2 flop + 2 discards
    } else {
      // Turn (street 4 -> 5) or River (street 5 -> 6)
      state.street = state.street + 1;
      state.player_id = 1;  // Player 1 acts first
      state.num_board_cards = state.street - 1;  // 5 for turn, 6 for river
    }
    
    return state;
  }

  // POKER HAND EVALUATION
  // Convert card index to rank and suit
  static int card_rank(int card) { return card / 4; }  // 0-12 (2-A, where 0=2, 12=Ace)
  static int card_suit(int card) { return card % 4; }  // 0-3 (clubs, diamonds, hearts, spades)
  
  // Get rank name for debugging
  static std::string rank_name(int rank) {
    const char* names[] = {"2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"};
    return names[rank];
  }
  
  // Get suit name for debugging
  static std::string suit_name(int suit) {
    const char* names[] = {"c", "d", "h", "s"};  // clubs, diamonds, hearts, spades
    return names[suit];
  }

  // Evaluate the best 5-card poker hand from available cards
  // Returns a hand rank value (higher = better hand)
  // Hand encoding: bits 0-12 = high card rank, bits 13-15 = hand type
  // Hand types: 0=high card, 1=pair, 2=two pair, 3=trips, 4=straight, 
  //             5=flush, 6=full house, 7=quads, 8=straight flush
  static int64_t evaluate_5card_hand(const std::vector<int>& cards);
  
  // Evaluate best hand from 2 hole cards + board cards (up to 6 board cards)
  // Returns hand rank value
  static int64_t evaluate_best_hand(const std::vector<int>& hole_cards, 
                                    const std::vector<int>& board_cards);
  
  // Compare two hand rank values (returns >0 if hand1 > hand2, <0 if hand1 < hand2, 0 if equal)
  static int compare_hands(int64_t hand1_rank, int64_t hand2_rank);
  
  // Get hand type from rank value
  static int get_hand_type(int64_t rank);
  
  // Convert hand index to actual card indices
  // hand is an index into all possible 3-card combinations
  // Returns vector of 3 card indices (0-51)
  static std::vector<int> hand_to_cards(int hand);
  
  // Get post-discard hole cards from pre-discard hand and discard choice
  // pre_discard_hand: index of 3-card hand
  // discard_index: which card to discard (0, 1, or 2)
  // Returns vector of 2 card indices
  static std::vector<int> get_post_discard_cards(int pre_discard_hand, int discard_index);

  std::string action_to_string(Action action) const;
  std::string state_to_string(const PartialPublicState& state) const;
  std::string action_to_string_short(Action action) const;
  std::string state_to_string_short(const PartialPublicState& state) const;

 private:
  // Compute number of possible 3-card hands: C(52, 3) = 22,100
  static int compute_num_hands() {
    // C(52, 3) = 52 * 51 * 50 / (3 * 2 * 1) = 22,100
    return (52 * 51 * 50) / 6;
  }

  static int int_pow(int base, int power) {
    if (power == 0) return 1;
    const int half_power = int_pow(base, power / 2);
    const int reminder = (power % 2 == 0) ? 1 : base;
    const double double_half_power = half_power;
    const double double_result =
        double_half_power * double_half_power * reminder;
    assert(double_result + 1 <
           static_cast<double>(std::numeric_limits<int>::max()));
    return half_power * half_power * reminder;
  }

  static constexpr int kInitialAction = -1;
  const int num_hands_;
  const Action num_actions_;
};

}  // namespace poker