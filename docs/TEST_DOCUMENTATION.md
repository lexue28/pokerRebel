# Test Documentation

## Purpose of Tests

The tests in `csrc/poker/` are **critical for verifying the poker variant works correctly** before training. They do NOT directly help with training, but they **catch bugs early** that would cause training to fail or produce incorrect results.

## Test Files

### 1. `poker_test.cc` - Core Game Logic Tests
**Purpose**: Verifies basic poker game mechanics work correctly.

**Tests**:
- **Action Unpacking**: Verifies that actions (Fold, Call, Bet, Discard) are correctly encoded/decoded
- **Initial State**: Checks that the game starts in the correct state (preflop, player 0 acts first)
- **State Transitions**: Verifies state changes after actions
- **Hand Evaluation**: Tests poker hand ranking (high card, pair, two pair, trips, etc.)
- **Hand-to-Cards Conversion**: Verifies that hand indices correctly map to 3-card combinations
- **Post-Discard Cards**: Tests that discarding a card from a 3-card hand produces the correct 2-card hand
- **Best Hand Evaluation**: Tests finding the best 5-card hand from hole cards + board

**Why Important**: If these fail, the game logic is broken and training will produce garbage.

### 2. `tree_test.cc` - Game Tree Traversal Tests
**Purpose**: Verifies the game tree unrolling (BFS traversal) works correctly.

**Tests**:
- **Tree Unrolling**: Tests that the game tree can be unrolled to various depths
- **Tree Structure**: Verifies parent-child relationships and depth tracking
- **Breadth-First Ordering**: **CRITICAL** - Verifies tree is in BFS order (required for partial initialization in solver)
- **State Equality**: Tests that state comparison works correctly

**Why Important**: The solver relies on BFS-ordered trees. If this is wrong, the solver will fail.

### 3. `recursive_solving_test.cc` - Solver Framework Tests
**Purpose**: Verifies the recursive solving components work.

**Tests**:
- **RlRunner**: Tests that the RL runner can step through games
- **Strategy Computation**: Verifies that strategies can be computed recursively
- **Zero Net**: Tests with a zero-value network (baseline)

**Why Important**: These verify the solver framework works. If these fail, training won't work.

### 4. `subgame_solving_test.cc` - Subgame Solver Tests
**Purpose**: Verifies subgame solving (CFR/FP) and query serialization work correctly.

**Tests**:
- **Hand Evaluation**: Tests all poker hand types (high card, pair, two pair, trips, straight, flush, full house)
- **Win Probability**: Tests computing win probabilities for terminal states
- **Fictitious Play (FP)**: Tests FP solving on small subgames
- **CFR**: Tests CFR solving on small subgames
- **Query Serialization**: **CRITICAL** - Tests that game states and beliefs can be serialized to/from float arrays for neural network input/output

**Why Important**: 
- Query serialization is how the C++ solver communicates with the Python neural network. If this is broken, training will fail.
- Hand evaluation must be correct or the solver will compute wrong values.

## What Happens If Tests Fail?

- **Training will likely fail** or produce incorrect results
- **Bugs will be caught early** instead of wasting hours/days of training time
- **Debugging is easier** with failing tests pointing to specific issues

## Running Tests

```bash
# Build tests
make

# Run all tests
make test

# Run specific test
./build/poker_game_test
./build/poker_tree_test
./build/poker_recursive_solving_test
./build/poker_subgame_solving_test
```

## Test Maintenance

When modifying poker game logic:
1. **Update tests** to match new behavior
2. **Add new tests** for new features
3. **Fix broken tests** - they're usually pointing to real bugs

## Card Encoding Reference

Cards are encoded as 0-51:
- `rank = card / 4` (0-12, where 0=2, 12=Ace)
- `suit = card % 4` (0=clubs, 1=diamonds, 2=hearts, 3=spades)

Examples:
- Card 0 = 2♣ (rank 0, suit 0)
- Card 1 = 2♦ (rank 0, suit 1)
- Card 4 = 3♣ (rank 1, suit 0)
- Card 16 = 6♣ (rank 4, suit 0)
