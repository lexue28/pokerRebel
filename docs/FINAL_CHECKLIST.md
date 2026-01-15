# Final Pre-Slurm Checklist

## ✅ Build System

- [x] **Makefile** points to `csrc/poker` (not `csrc/liars_dice`)
- [x] **CMakeLists.txt** builds `poker_lib` (not `liars_dice_lib`)
- [x] All source files use `poker::` namespace (not `liars_dice::`)
- [x] All test targets renamed to `poker_*_test`

## ✅ C++ Code

- [x] **Game class** takes no parameters: `Game()` (not `Game(num_dice, num_faces)`)
- [x] **PartialPublicState** has `operator==` and `operator!=`
- [x] **RecursiveSolvingParams** doesn't require `num_dice`/`num_faces`
- [x] All `unroll_tree()` calls use correct signature
- [x] **Query size** matches: `1 + 1 + 15 + 6 + 2 + 1 + 22100*2 = 44,226`
- [x] **Num actions** = 15 (1 fold + 1 call/check + 10 bet sizes + 3 discards)
- [x] **Num hands** = 22,100 (C(52,3))

## ✅ Python Code

- [x] **Net2Poker** model exists in `cfvpy/models.py`
- [x] **input_size_poker()** returns 44,226
- [x] **output_size_poker()** returns 22,100
- [x] **Poker detection** in `_build_model()` checks `env.game == 'toss_holdem'`
- [x] **num_actions** set to 15 for poker in `CFVExp.__init__`
- [x] **GPU check** relaxed for CPU-only data generation

## ✅ Configuration

- [x] **poker.yaml** has `env.game: toss_holdem`
- [x] **poker.yaml** doesn't have `env.num_dice` or `env.num_faces`
- [x] **slurm_mit.yaml** exists with correct partition

## ✅ Tests

- [x] All tests use `poker::` namespace
- [x] All tests use `Game()` constructor (no parameters)
- [x] Hand evaluation tests use correct card indices
- [x] `operator!=` added to `PartialPublicState`
- [x] `RlRunner` constructor uses `RecursiveSolvingParams`

## ✅ Critical Functions

- [x] **get_query_size()** returns 44,226
- [x] **write_query_to()** serializes state correctly
- [x] **deserialize_query()** deserializes correctly
- [x] **hand_to_cards()** converts hand index to cards
- [x] **evaluate_5card_hand()** evaluates poker hands correctly

## 🚀 Slurm Command

```bash
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    launcher=slurm_mit \
    launcher.num_gpus=2 \
    env.subgame_params.use_cfr=true \
    selfplay.cpu_gen_threads=0 \
    selfplay.threads_per_gpu=8
```

## 🔍 What to Check in Logs

When the job starts, look for these in the first 50 lines:

1. ✅ `POKER VARIANT DETECTED: Using Net2Poker model`
2. ✅ `Built a model: Net2Poker(...)`
3. ✅ `POKER VARIANT: num_actions = 15`
4. ✅ No errors about `num_dice` or `num_faces`
5. ✅ No errors about `liars_dice` namespace

## ⚠️ Common Build Errors (Already Fixed)

- ❌ `'poker::Game' is not a base of...` → Fixed: Changed `tree[node_id].Game::get_active_player()` to `game.get_active_player(tree[node_id].state)`
- ❌ `redeclaration of 'int active'` → Fixed: Removed duplicate declaration
- ❌ `'discard_complete' has no member` → Fixed: Changed to `discard_choice[0]` and `discard_choice[1]`
- ❌ `no match for 'operator!='` → Fixed: Added `operator!=` to `PartialPublicState`
- ❌ `no matching function for call to 'RlRunner::RlRunner'` → Fixed: Use `RecursiveSolvingParams` instead of `SubgameSolvingParams`

## 📋 Build Steps on Cluster

```bash
# 1. Transfer code (from local machine)
scp -r rebel/ lexue28@login007.mit.edu:~/rebel

# 2. SSH and setup
ssh lexue28@login007.mit.edu
cd ~/rebel
conda activate rebel

# 3. Build
make clean
make

# 4. Run training
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    launcher=slurm_mit \
    launcher.num_gpus=2 \
    env.subgame_params.use_cfr=true \
    selfplay.cpu_gen_threads=0 \
    selfplay.threads_per_gpu=8
```

## 🎯 Verification

Before running, verify:
- [ ] `make` completes without errors
- [ ] All tests pass: `make test`
- [ ] Python can import: `python -c "import cfvpy.models; print('OK')"`
- [ ] Config loads: `python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('conf/c02_selfplay/poker.yaml'); print(cfg.env.game)"`
