#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=sg_guidance_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=%x.%A.out
#SBATCH --error=%x.%A.err

# Space-group guidance sweep — the CFG "did conditioning work?" payoff loop.
#
# For a conditional (space-group LoRA) checkpoint, generates NUM_SAMPLES crystals
# per (target space group x guidance weight w), relaxes them with NequIP batch
# relaxation by default, and measures the exact space-group match-rate vs the
# target (pymatgen SpacegroupAnalyzer), plus lift over the shared unconditional
# base pool. Writes sweep.csv / summary.json / a
# match_rate_vs_w.png to OUT_DIR.
#
# Protocol defaults follow the literature so the number is comparable:
#   targets = 10 most common MP-20 SGs (2 12 14 62 63 139 166 194 221 225)
#   guidance grid brackets MatterGen's gamma=2; headline symprec = 0.1 A.
#
# Submit (after finetune_sg_conditioning.slurm):
#   sbatch --export=ALL,CHECKPOINT=outputs/sg_cond/checkpoints/best.pt,\
# OUT_DIR=results/sg_sweep scripts/sg_guidance_sweep.slurm

set -euo pipefail

module purge
module load 2024
module load CUDA/12.6.0

export PATH="$HOME/.local/bin:$PATH"

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/crystalite}"
CRYSFINITY_ROOT="${CRYSFINITY_ROOT:-$HOME/crysfinity}"
cd "$PROJECT_ROOT"

export UV_CACHE_DIR="${TMPDIR:-/tmp/$USER/uv-cache-${SLURM_JOB_ID:-manual}}"
mkdir -p "$UV_CACHE_DIR"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a conditional (space-group) checkpoint}"
OUT_DIR="${OUT_DIR:-results/sg_sweep}"
DATA_ROOT="${DATA_ROOT:-$CRYSFINITY_ROOT/data/mp20}"
DATASET_NAME="${DATASET_NAME:-mp20}"
NMAX="${NMAX:-20}"

NUM_SAMPLES="${NUM_SAMPLES:-1000}"
SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-100}"
SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-256}"
SAMPLE_SEED="${SAMPLE_SEED:-1234}"
SAMPLE_MODE="${SAMPLE_MODE:-ema}"

# ---- relaxation before SG analysis ----
RELAX_MLIP="${RELAX_MLIP:-nequip}"
RELAX_STEPS="${RELAX_STEPS:-200}"
RELAX_BATCH_SIZE="${RELAX_BATCH_SIZE:-64}"
RELAX_DEVICE="${RELAX_DEVICE:-cuda}"
NEQUIP_RELAX_MODE="${NEQUIP_RELAX_MODE:-batch}"
NEQUIP_OPTIMIZER="${NEQUIP_OPTIMIZER:-FIRE}"
NEQUIP_CELL_FILTER="${NEQUIP_CELL_FILTER:-frechet}"
NEQUIP_FMAX="${NEQUIP_FMAX:-0.005}"
NEQUIP_MAX_FORCE_ABORT="${NEQUIP_MAX_FORCE_ABORT:-1000000}"
NEQUIP_COMPILE_PATH="${NEQUIP_COMPILE_PATH:-}"
if [[ -z "$NEQUIP_COMPILE_PATH" ]]; then
  if [[ -d "$PROJECT_ROOT/mlips" ]]; then
    NEQUIP_COMPILE_PATH="$PROJECT_ROOT/mlips"
  elif [[ -d "$CRYSFINITY_ROOT/mlips" ]]; then
    NEQUIP_COMPILE_PATH="$CRYSFINITY_ROOT/mlips"
  else
    NEQUIP_COMPILE_PATH="$PROJECT_ROOT/data/mlip/nequip"
  fi
fi

# 10 most common MP-20 space groups; guidance grid brackets MatterGen gamma=2.
TARGETS="${TARGETS:-2 12 14 62 63 139 166 194 221 225}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-0 1 2 4 8}"
SYMPRECS="${SYMPRECS:-0.01 0.05 0.1 0.3}"
HEADLINE_SYMPREC="${HEADLINE_SYMPREC:-0.1}"

mkdir -p "$OUT_DIR"

echo "[sg-sweep] checkpoint=$CHECKPOINT out_dir=$OUT_DIR"
echo "[sg-sweep] project_root=$PROJECT_ROOT crysfinity_root=$CRYSFINITY_ROOT data_root=$DATA_ROOT"
echo "[sg-sweep] relaxation=$RELAX_MLIP mode=$NEQUIP_RELAX_MODE steps=$RELAX_STEPS batch=$RELAX_BATCH_SIZE nequip=$NEQUIP_COMPILE_PATH"
echo "[sg-sweep] targets=[$TARGETS] guidance=[$GUIDANCE_SCALES] n/cell=$NUM_SAMPLES"

srun uv run python src/eval/sg_guidance_sweep.py \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUT_DIR" \
  --data_root "$DATA_ROOT" \
  --dataset_name "$DATASET_NAME" \
  --nmax "$NMAX" \
  --bf16 \
  --sample_mode "$SAMPLE_MODE" \
  --num_samples "$NUM_SAMPLES" \
  --sample_num_steps "$SAMPLE_NUM_STEPS" \
  --sample_chunk_size "$SAMPLE_CHUNK_SIZE" \
  --sample_seed "$SAMPLE_SEED" \
  --relax_mlip "$RELAX_MLIP" \
  --relax_steps "$RELAX_STEPS" \
  --relax_batch_size "$RELAX_BATCH_SIZE" \
  --relax_device "$RELAX_DEVICE" \
  --nequip_compile_path "$NEQUIP_COMPILE_PATH" \
  --nequip_relax_mode "$NEQUIP_RELAX_MODE" \
  --nequip_optimizer "$NEQUIP_OPTIMIZER" \
  --nequip_cell_filter "$NEQUIP_CELL_FILTER" \
  --nequip_fmax "$NEQUIP_FMAX" \
  --nequip_max_force_abort "$NEQUIP_MAX_FORCE_ABORT" \
  --targets $TARGETS \
  --guidance_scales $GUIDANCE_SCALES \
  --symprecs $SYMPRECS \
  --headline_symprec "$HEADLINE_SYMPREC"

echo "[sg-sweep] done -> $OUT_DIR/sweep.csv, summary.json, match_rate_vs_w.png"
