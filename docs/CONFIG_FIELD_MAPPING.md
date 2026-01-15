# Config Field Mapping: Python ↔ C++

## Overview

This document tracks which Python config fields map to C++ structs and which are Python-only.

## RecursiveSolvingParams

**Python Config (`env` section):**
- `game: toss_holdem` - **Python-only** (used for model detection)
- `num_dice: 1` - **Python-only** (dice variant only)
- `num_faces: 4` - **Python-only** (dice variant only)
- `random_action_prob: 0.25` - ✅ Maps to C++
- `sample_leaf: true` - ✅ Maps to C++
- `subgame_params: {...}` - ✅ Maps to C++ (nested struct)

**C++ Struct (`poker::RecursiveSolvingParams`):**
- `float random_action_prob`
- `bool sample_leaf`
- `SubgameSolvingParams subgame_params`

## SubgameSolvingParams

**Python Config (`env.subgame_params` section):**
- `num_iters: 256` - ✅ Maps to C++
- `max_depth: 4` - ✅ Maps to C++
- `linear_update: true` - ✅ Maps to C++
- `use_cfr: false` - ✅ Maps to C++
- `optimistic: false` - ✅ Maps to C++
- `dcfr: false` - ✅ Maps to C++
- `dcfr_alpha: 0.0` - ✅ Maps to C++
- `dcfr_beta: 0.0` - ✅ Maps to C++
- `dcfr_gamma: 0.0` - ✅ Maps to C++

**C++ Struct (`poker::SubgameSolvingParams`):**
- `int num_iters`
- `int max_depth`
- `bool linear_update`
- `bool use_cfr`
- `bool optimistic`
- `bool dcfr`
- `double dcfr_alpha`
- `double dcfr_beta`
- `double dcfr_gamma`

## Filtering Logic

The `create_mdp_config()` function in `cfvpy/selfplay.py` automatically filters out Python-only fields before passing config to C++:

```python
python_only_fields = {'game', 'num_dice', 'num_faces'}
filtered_dict = {k: v for k, v in cfg_dict.items() if k not in python_only_fields}
```

## Error Prevention

If a new Python-only field is added:
1. Add it to `python_only_fields` set in `create_mdp_config()`
2. Update this document
3. Ensure it's not accidentally passed to C++

If a field mismatch error occurs:
- Check if the field is Python-only and should be filtered
- Check if the field exists in the C++ struct but isn't exposed via pybind11
- Check the error message for available fields
