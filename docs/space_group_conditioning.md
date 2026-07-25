# Space-Group-Conditioned Generation in Crystalite

## What we are doing

Crystalite's DNG model (`dng_clean.pt`) generates crystal structures
**unconditionally** — it samples plausible crystals but you cannot ask it for a
specific property. We are adding **controllable generation conditioned on space
group**: the ability to say "generate a crystal in space group 225" and have the
model steer toward that symmetry.

We do this **without retraining the base model**. We freeze `dng_clean.pt`,
attach a small space-group embedding plus **LoRA adapters** to the model's
conditioning pathway, and finetune only those (a few percent of the parameters)
on the existing MP-20 labels. At sampling time we use **classifier-free guidance
(CFG)** to control how strongly the space-group condition is enforced.

## Why space group (and not band gap or formation energy)

The property is chosen for **verifiability**, not glamour. To claim conditioning
"works," we must measure the *true* property of a **generated** structure — one
that has no DFT label — and compare it to the target. For most properties that
measurement is expensive or approximate:

| Property | How to verify a generated structure | Cost |
|---|---|---|
| **Space group** | one deterministic `pymatgen` symmetry call | **free, exact** |
| Formation energy | ML-potential (CHGNet) relax + energy | slow, approximate |
| Band gap | DFT or a separate ML predictor | expensive, noisy |

Space group is a **geometric property fully determined by the coordinates we
generate**, so it gives the tightest possible "did conditioning work?" loop: no
DFT, no surrogate model, no relaxation. That makes it the ideal first target for
demonstrating a conditioning mechanism.

## How it works

The trunk is a **DiT-style transformer with AdaLN-Zero** modulation
(`src/models/transformer.py`, `AdaLNBlock`). The entire conditioning signal
enters through **one vector** — the time embedding `t_emb`
(`src/crystalite/crystalite.py`) — which every block turns into per-layer
scale/shift/gate. This single, well-defined injection point is what makes
lightweight conditioning clean:

1. **PropEncoder** (`src/models/embeddings.py`): an embedding table over space
   groups, `nn.Embedding(231, d_model)`, index 0 reserved as the *null token*.
   Its output is added to `t_emb`. Zero-initialised, so at the start of finetuning
   the conditioning is a no-op and the model is exactly the frozen base.

2. **LoRA on AdaLN** (`src/models/lora.py`): each block's modulation `Linear` is
   wrapped with a low-rank adapter (zero-initialised, so it starts as an exact
   no-op). The base weights stay frozen; only the adapters + PropEncoder train.

3. **Condition-dropout** (`src/train_crystalite.py`): during finetuning the target
   space group is randomly replaced by the null token (p ≈ 0.15), so the model
   learns *both* the conditional and unconditional score — the prerequisite for CFG.

4. **Classifier-free guidance** (`src/crystalite/sampler.py`, `denoise_cfg`): at
   sampling we run the denoiser twice (target token and null token) and extrapolate
   on the predicted clean structure,
   `D = D_null + w·(D_cond − D_null)`,
   where `w` is the guidance weight (`w=0` unconditional, `w=1` pure conditional,
   `w>1` stronger). Fractional coordinates are combined with the minimum-image
   convention because they live on a torus.

Because the base is frozen and the adapters start as no-ops, the *unconditional*
prior is inherited exactly from `dng_clean.pt` — we only learn the conditioning
delta, on a few percent of the parameters, from the 27k labelled MP-20 crystals.

## Relation to prior work

This is, deliberately, the **MatterGen** recipe (Zeni et al., *Nature* 2025):
a pretrained crystal diffusion model, conditioned via **lightweight adapter
modules injected additively into the layers**, finetuned with **classifier-free
guidance and condition-dropout**, targeting space-group symmetry among other
properties. Our contribution is the **parameter-efficient LoRA variant** on the
AdaLN pathway and the Crystalite/EDM base — not the paradigm itself.

There are two families in the literature, and we are firmly in the first:

- **Soft conditioning** (MatterGen, CrystalFormer, *this work*): condition and
  measure how often you hit the target. Match-rates are modest.
- **Hard constraint** (DiffCSP++, SymmCD, Wyckoff-based models): build the space
  group in by construction so it is guaranteed. These get ~100% by design but
  require architectural changes, not a conditioning add-on.

## How we evaluate

For each target space group and guidance weight `w`, we generate `N` crystals and
compute each one's space group with `pymatgen`'s `SpacegroupAnalyzer`
(`src/eval/spacegroup_match.py`), then report the **exact match-rate** to the
target. The evaluation protocol follows the literature so the number is directly
comparable:

- **Targets**: the 10 most common MP-20 space groups (`2, 12, 14, 62, 63, 139,
  166, 194, 221, 225`) — the SymmCD "10 SGs" set.
- **Tolerance**: spglib `symprec = 0.1 Å` as the headline (the MatterGen/SymmCD
  standard), with a sweep `{0.01, 0.05, 0.1, 0.3}` for robustness, because
  generated structures have sub-Ångström noise and symmetry detection is
  tolerance-sensitive.
- **Guidance grid**: `w ∈ {0, 1, 2, 4, 8}`; `w=2` matches MatterGen's `γ=2`.
- **Headline metrics**: (a) the **match-rate-vs-`w` curve** (the CFG story), and
  (b) **lift over the unconditional base rate** — match-rate divided by how often
  *unconditioned* generation lands in that space group. Lift controls for the fact
  that common space groups are hit often by chance, and it cancels the arbitrariness
  of the symprec choice (a loose tolerance inflates both terms equally).

**Expectation.** Soft conditioning on symmetry is genuinely hard: MatterGen
recovers the target space group only **~20% of the time (~10% for high-symmetry
groups like 221/225)**. So success is framed as *lift over base rate* and a
*head-to-head against MatterGen's numbers*, not as an absolute hit-rate threshold.

## How to run it

```bash
# 1. Finetune the conditioning adapters (Snellius; arch pinned to dng_clean.pt)
sbatch --export=ALL,OUT_DIR=outputs/sg_cond,WANDB_NAME=sg_cond \
  scripts/finetune_sg_conditioning.slurm

# 2. Sweep guidance and produce the match-rate-vs-w curve
sbatch --export=ALL,CHECKPOINT=outputs/sg_cond/checkpoints/best.pt,OUT_DIR=results/sg_sweep \
  scripts/sg_guidance_sweep.slurm
```

The sweep writes `sweep.csv`, `summary.json`, and `match_rate_vs_w.png` to
`OUT_DIR`. The headline result is match-rate at `symprec 0.1`, `w=2` — the number
directly comparable to MatterGen.

## Results

_To be filled after the Snellius run._ Headline table: exact space-group
match-rate (%) at `symprec = 0.1`, one row per target space group, one column per
guidance weight. `w=0` is the unconditional **base rate**; `best lift` is the
largest `match_rate(w) / base_rate` across the guidance grid. Numbers come
straight from `results/sg_sweep/sweep.csv`.

| Target SG | base (w=0) | w=1 | w=2 | w=4 | w=8 | best lift |
|---|---|---|---|---|---|---|
| 2 — P-1 | — | — | — | — | — | — |
| 12 — C2/m | — | — | — | — | — | — |
| 14 — P2₁/c | — | — | — | — | — | — |
| 62 — Pnma | — | — | — | — | — | — |
| 63 — Cmcm | — | — | — | — | — | — |
| 139 — I4/mmm | — | — | — | — | — | — |
| 166 — R-3m | — | — | — | — | — | — |
| 194 — P6₃/mmc | — | — | — | — | — | — |
| 221 — Pm-3m | — | — | — | — | — | — |
| 225 — Fm-3m | — | — | — | — | — | — |
| **mean** | — | — | — | — | — | — |

Comparison point: MatterGen reports ~20% target-SG match (~10% for high-symmetry
groups such as 221/225) at the same `symprec = 0.1` tolerance. `w=2` is the
directly comparable setting. The CFG curve (`match_rate_vs_w.png`) and the
per-target lift are the supporting evidence.

## Key files

| File | Role |
|---|---|
| `src/models/lora.py` | LoRA adapter + AdaLN wrapping + base-checkpoint key remap |
| `src/models/embeddings.py` | `PropEncoder` (space-group embedding, null token) |
| `src/crystalite/crystalite.py` | conditioning injection into `t_emb` |
| `src/crystalite/sampler.py` | `denoise_cfg` (classifier-free guidance) |
| `src/train_crystalite.py` | base load, freeze, condition-dropout |
| `src/eval/spacegroup_match.py` | space-group match-rate metric |
| `src/eval/sg_guidance_sweep.py` | end-to-end guidance sweep driver |
| `scripts/finetune_sg_conditioning.slurm` | finetune job (Snellius) |
| `scripts/sg_guidance_sweep.slurm` | evaluation sweep job (Snellius) |

## References

- Zeni et al., *A generative model for inorganic materials design* (MatterGen),
  Nature 2025 — adapter finetuning + CFG for property-conditioned crystals.
- Jiao et al., *Space Group Constrained Crystal Generation* (DiffCSP++), ICLR 2024.
- Levy, Panigrahi et al., *SymmCD: Symmetry-Preserving Crystal Generation with
  Diffusion Models*, 2025.
