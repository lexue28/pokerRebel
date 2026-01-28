# MIT Pokerbots 2026
This repository is built upon Facebook's ReBeL (cited below) for the [MIT Pokerbots 2026 competition](https://github.com/mitpokerbots/class-resources-2026). The code was adapted from Liar's Dice to a poker variant with a three flop and trained using MIT's Engaging Cluster, then submitted to the scrimmage server.


# ReBeL

Implementation of [ReBeL](https://arxiv.org/abs/2007.13544) for a poker [variant](https://github.com/mitpokerbots/class-resources-2026/blob/main/variant.pdf), an algorithm that generalizes the paradigm of self-play reinforcement learning and search to imperfect-information games.

## Installation

The recommended way to install ReBeL is via conda env.

First, clone and create the conda env:

```bash
git clone --recursive https://github.com/lexue28/rebel
cd rebel
conda create --yes -n rebel python=3.7
source activate rebel
```

Then, install dependencies:

```bash
pip install -r requirements.txt
conda install cmake
```

Finally, compile the C++ part:

```bash
make
```

## Training a value net

Use the batch scripts in /batch such as [batch/run_cpu_batch_balanced.sh](batch/run_cpu_batch_balanced.sh) or the following command to train a value net with data generation placed on CPU:

```
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    env.num_dice=1 \
    env.num_faces=4 \
    env.subgame_params.use_cfr=true \
    selfplay.cpu_gen_threads=60
```

Check the config [conf/c02_selfplay/poker.yaml](conf/c02_selfplay/poker.yaml) for all possible parameters. If use use Slurm to manage the cluster, add `launcher=slurm_8gpus launcher.num_gpus=NUM_GPUS` to run the job on the cluster with GPU. If you specify `NUM_GPUS > 8`, the code will assume that you are launching on several machines with 8 GPUs each.


## Evaluating a value net

The trainer saves checkpoints every 10 epochs as state dictionaries and as TorchScript modules. You can use the latter to compute exploitability of strategy produced with such a model using the following command:

```
build/recursive_eval \
    --net path/to/model.torchscript \
    --subgame_iters 1024 \
    --num_repeats 4097 \
    --num_threads 10 \
    --cfr
```

Setting `--num_repeats` to a positive value enables evaluation of a sampled policy, i.e., when we use a randomly selected iteration of the underlying subgame algorithm for the subgame. Computing the exact full policy produced by such a process is intractable. Therefore, we average `num_repeats` such policies to get an upper bound for the exploitability.

The script reports exploitability for both full tree solving and recursive solving.

## Code structure

The training loop is implemented in Python and located in [cfvpy/selfplay.py](cfvpy/selfplay.py). The actual data generation part happens in C++ and could be found in [csrc/poker](csrc/poker).

## Citation

```bibtex
@article{brown2020rebel,
  title={Combining deep reinforcement learning and search for imperfect-information games},
  author={Brown, Noam and Bakhtin, Anton and Lerer, Adam and Gong, Qucheng},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
