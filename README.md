<p align="center">
  <img src="media/Crystalite logo-03.png" alt="Crystalite" width="400" />
</p>

---
>[!warning] Note
>This is a pre-release of the `crystalite` codebase and is not necessarily representative of the final working product.


`crystalite` is a codebase for tokenized crystal representations, EDM-based generation, and evaluation for two workflows:

- DNG: de novo generation of atom types, fractional coordinates, and lattice parameters
- CSP: crystal structure prediction with atom types fixed from a target composition/structure

Both workflows use the same main training entrypoint, `src/train_crystalite.py`, and diverge by flags and evaluation metrics.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Data and Representation](#data-and-representation)
  - [Atom representations](#atom-representations)
- [Training Workflows](#training-workflows)
  - [DNG training](#dng-training)
  - [CSP training](#csp-training)
- [Evaluation Workflows](#evaluation-workflows)
  - [DNG train-time evaluation](#dng-train-time-evaluation)
  - [CSP train-time evaluation](#csp-train-time-evaluation)
  - [DNG post-training checkpoint evaluation](#dng-post-training-checkpoint-evaluation)
  - [Offline checkpoint sampling](#offline-checkpoint-sampling)
  - [CSP post-training evaluation](#csp-post-training-evaluation)
- [Thermo Backends](#thermo-backends)
  - [Phase diagram (hull)](#phase-diagram-hull)
  - [NequIP OAM-L setup](#nequip-oam-l-setup)
- [Outputs and Artifacts](#outputs-and-artifacts)

## Project Overview

The current model stack is an EDM sampler paired with a transformer trunk and optional GEM-based geometry attention bias. GEM stands for Geometry Enhancement Module: it injects learned distance and/or edge-conditioned attention bias into the transformer when `--use_distance_bias` or `--use_edge_bias` is enabled. The repo is centered on MP20-style tokenized crystal data plus evaluation utilities for:

- train-time sampling metrics
- post-training DNG checkpoint evaluation
- optional thermo stability checks with CHGNet or NequIP

At a glance:

| Mode | Enable with | Predicts | Main train-time evaluation |
| --- | --- | --- | --- |
| DNG | default | atom types + coords + lattice | validity, Wasserstein, novelty/UN, SUN/MSUN, thermo |
| CSP | `--csp` | coords + lattice with fixed atom types | match rate, RMS, optional precise top-k |

Advanced ablations, grids, and post-hoc utilities live under `scripts/`. This README focuses on the current main workflows rather than exhaustively documenting those scripts.

## Environment Setup

The repo targets Python 3.12 and is set up with `uv`.

```bash
uv python install 3.12
uv sync
```

Notes:

- `pyproject.toml` pins PyTorch/Torchvision through `uv` indexes: CPU wheels outside Linux, CUDA 12.8 wheels on Linux.
- `uv sync` installs both CHGNet and the NequIP/TorchSim stack, but NequIP thermo evaluation still requires a compiled `.nequip.pt2` model if you choose `--thermo_mlip nequip`.
- Use `uv sync --group dev` if you also want the dev dependencies such as `pytest`.

## Data and Representation

The core dataset class is `src.data.mp20_tokens.MP20Tokens`. Each structure is represented as:

- `A0`: `(NMAX,)` atom types, padded with `0`
- `F1`: `(NMAX, 3)` fractional coordinates
- `Y1`: `(6,)` lattice representation
- `pad_mask`: `(NMAX,)`, `True` where padded

Minimal example:

```python
from src.data.mp20_tokens import MP20Tokens

ds = MP20Tokens(
    root="data/mp20",
    augment_translate=True,
    split="train",
    nmax=20,
)
item = ds[0]
```

Dataset presets are selected with `--dataset_name` and currently include `mp20`, `mpts_52`,  `alex_mp20`, and `custom`. `--nmax` is resolved from the dataset preset unless overridden explicitly.

Raw dataset files for `mpts_52`, `alex_mp20`, and `mp20` are available from the HuggingFace dataset repository:

> https://huggingface.co/datasets/jbungle/crystalite-datasets/tree/main

`mp20` can also be auto-downloaded from `chaitjo/MP20_ADiT` on first use and does not need to be fetched manually.

### Atom representations

For the atom-type channel, the current training code supports multiple element representations through `--type_encoding`:

- `atomic_number`: direct element-channel encoding, used by default
- `periodic_table_2d`: CrystalFlow-style row/column periodic-table encoding
- `electron_config`: ground-state electron-configuration features
- `chem_raw_v2` and `chem_pca_v2_*`: hand-crafted chemistry descriptors, either raw or PCA-compressed

These representations affect how atom types are encoded and decoded inside the EDM model. In DNG mode they shape the sampled atom-type path; in CSP mode atom types are fixed, but the chosen representation still determines the internal type features seen by the model.

## Training Workflows

### DNG training

Canonical DNG run:

```bash
python src/train_crystalite.py \
  --data_root data/mp20 \
  --dataset_name mp20 \
  --output_dir outputs/dng_mp20 \
  --sample_every 1000 \
  --sample_metrics_count 2048 \
  --precise_every 50000 \
  --precise_metrics_count 10000 \
  --best_ckpt
```

Behavior:

- atom types, coordinates, and lattice are all sampled
- train-time sampling logs DNG metrics
- `--best_ckpt` tracks the best checkpoint using the configured DNG metric policy

### CSP training

Canonical CSP run:

```bash
python src/train_crystalite.py \
  --csp \
  --data_root data/mp20 \
  --dataset_name mp20 \
  --output_dir outputs/csp_mp20 \
  --sample_every 1000 \
  --sample_metrics_count 256 \
  --precise_every 50000 \
  --best_ckpt
```

Behavior:

- atom types are fixed from the target structures
- CSP mode zeroes the type-loss path and evaluates reconstruction quality instead of DNG novelty metrics
- precise CSP sampling can report best-of-k metrics via `--csp_precise_topk_list` and `--csp_precise_topk_samples`

## Evaluation Workflows

### DNG train-time evaluation

When sampling is enabled during training, DNG can log:

- validity metrics
- Wasserstein distribution distances
- novelty, unique+novel rate, and related DNG metrics
- SUN/MSUN if thermo relaxation is enabled and `sun_k` or `precise_sun_k` is positive
- standalone thermo metrics and generated-vs-reference thermo comparisons

These metrics are driven by the train-time sampling settings such as:

- `--sample_every`
- `--sample_metrics_count`
- `--precise_every`
- `--precise_metrics_count`
- `--sample_mode`
- `--sample_num_steps`

### CSP train-time evaluation

In CSP mode, train-time sampling logs reconstruction metrics rather than DNG novelty metrics:

- match rate
- mean RMS distance
- optional precise top-k metrics

### DNG post-training checkpoint evaluation

The first-class standalone checkpoint-eval entrypoint is `src/eval_subatomic_edm_ckpt.py`, which is currently DNG-oriented.

Canonical DNG checkpoint eval:

```bash
python src/eval_subatomic_edm_ckpt.py \
  --train_output_dir outputs/dng_mp20 \
  --checkpoint_preference best \
  --num_samples 10000 \
  --sample_mode ema
```

This path can compute:

- validity / composition / structure validity
- diagnostic metrics
- Wasserstein distribution metrics
- novelty / unique+novel metrics
- optional thermo metrics and SUN sample export

### Offline checkpoint sampling

If you only want to sample structures from a checkpoint without running the evaluator stack, use `src/sample_subatomic_edm_ckpt.py`.

Minimal offline sampling:

```bash
python src/sample_subatomic_edm_ckpt.py \
  --checkpoint outputs/dng_mp20/checkpoints/best.pt \
  --num_samples 256 \
  --output_dir outputs/dng_mp20/offline_demo
```

Behavior:

- loads the checkpoint directly and samples with regular or EMA weights
- writes `samples.pt` plus `samples.xyz` (extxyz) by default
- can optionally write per-sample CIFs with `--save_cifs`
- can run without dataset access; in that case atom counts fall back to `nmax` unless you pass `--fixed_num_atoms`

If the training dataset is available locally, the script can also reuse it for empirical atom-count sampling and train-split element masking:

```bash
python src/sample_subatomic_edm_ckpt.py \
  --checkpoint outputs/dng_mp20/checkpoints/best.pt \
  --num_samples 256 \
  --atom_count_strategy empirical \
  --data_root data/mp20 \
  --dataset_name mp20
```

### CSP post-training evaluation

There is no single first-class standalone CSP checkpoint-eval CLI analogous to `src/eval_subatomic_edm_ckpt.py` at the moment. Current standalone CSP evaluation is script-driven; for advanced preset or grid-style runs, see the relevant utilities under `scripts/`, for example:

- `scripts/grid_csp_inference.py`
- `scripts/eval_csp_mpts52_test_presets.py`

## Thermo Backends

Thermo stability is optional in both train-time sampling eval and DNG checkpoint eval.

- CHGNet is the default backend.
- NequIP is optional and requires `--thermo_mlip nequip` plus a valid `--nequip_compile_path`.
- In training, thermo metrics are enabled with `--thermo_stability_check`.
- In checkpoint eval, thermo metrics are enabled with `--thermo_count > 0`.

### Phase diagram (hull)

Both backends use a Materials Project phase diagram pickle for e-above-hull computation. Download `2023-02-07-ppd-mp.pkl` from Matbench Discovery v1.0.0 on Figshare:

> https://figshare.com/articles/dataset/Matbench_Discovery_v1_0_0/22715158?file=40344436

Place the file in `mp_02072023/`:

```
mp_02072023/
└── 2023-02-07-ppd-mp.pkl
```

Pass the path with `--ppd_path mp_02072023/2023-02-07-ppd-mp.pkl`, or set `thermo_cfg.ppd_path` when constructing `StabilityLogger` programmatically.

### NequIP OAM-L setup

The recommended NequIP model is [NequIP-OAM-L v0.1](https://www.nequip.net/models/mir-group/NequIP-OAM-L:0.1). It must be compiled to a `.nequip.pt2` file before use.

Compile for GPU (AOTInductor):

```bash
nequip-compile \
  mir-group/NequIP-OAM-L:0.1 \
  data/mlip/nequip/NequIP-OAM-L.nequip.pt2 \
  --mode aotinductor \
  --device cuda \
  --target ase
```

For CPU inference, replace `--device cuda` with `--device cpu`.

Place the compiled file anywhere under a directory matched by `--nequip_compile_path`. The default glob is `data/mlip/nequip/*.nequip.pt2`.

Optional NequIP-based DNG checkpoint eval:

```bash
python src/eval_subatomic_edm_ckpt.py \
  --train_output_dir outputs/dng_mp20 \
  --checkpoint_preference best \
  --num_samples 2048 \
  --thermo_count 256 \
  --thermo_mlip nequip \
  --nequip_compile_path "data/mlip/nequip/*.nequip.pt2"
```

If you want batched NequIP relaxation during train-time thermo eval, use `--thermo_mlip nequip --nequip_relax_mode batch`. That path is only valid for NequIP.

## Outputs and Artifacts

Training writes to `--output_dir` and creates:

- `checkpoints/`
- `samples/`

Common checkpoint artifacts include:

- `checkpoints/best.pt` when `--best_ckpt` is enabled
- `checkpoints/final.pt`
- `checkpoints/step_latest.pt` or step snapshots, depending on `--ckpt_every` and `--ckpt_latest_only`
- `checkpoints/epoch_latest.pt`

Sample artifacts are written under:

- `output_dir/samples/<tag>_step_<step>/...`

where `<tag>` is the sampling run tag such as `sample`, `sample_ema`, `precise`, or `precise_ema` and, in CSP mode, may include the split label.

DNG checkpoint eval writes reports under:

- `train_output_dir/eval_reports/<run_name>/metrics.json`

Optional checkpoint-eval artifacts include:

- `samples.pt` when `--save_samples_pt` is enabled
- `sun_samples/` and a manifest when `--save_sun_samples` is enabled

If W&B logging is enabled during training, metrics and rendered sample images are also logged there.

Logo design by [Dee Vasilevskaia](https://deevasilevskaia.com/).