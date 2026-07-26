#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=sg_cond_finetune
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=%x.%A.out
#SBATCH --error=%x.%A.err

# Space-group-conditioning finetune (LoRA on AdaLN + classifier-free guidance).
#
# Loads the frozen unconditional DNG checkpoint (dng_clean.pt), attaches a
# space-group PropEncoder + LoRA adapters on the AdaLN modulation, and finetunes
# ONLY those (base frozen) with condition-dropout so the model learns both the
# conditional and unconditional (null-token) score for CFG.
#
# The architecture block is pinned to dng_clean.pt's model_args (the M0 recipe:
# subatomic_tokenizer_pca_16, 16 heads / 14 layers, ltri lattice, edge-bias
# 256/12/32, loss 1/50/5, sigma_data 0.3, S_churn 60). These MUST match the base
# or the frozen-base load fails on a shape mismatch. Every flag is an env var.
#
# In-training DNG sampling/thermo metrics are OFF by default (SAMPLE_FREQUENCY=0)
# so this run needs no NequIP/phase-diagram artifacts; evaluate space-group
# effectiveness afterwards with scripts/sg_guidance_sweep.slurm.
#
# Submit:
#   sbatch --export=ALL,OUT_DIR=outputs/sg_cond,WANDB_NAME=sg_cond \
#     scripts/finetune_sg_conditioning.slurm
#   # from scratch data reprocess adds spacegroup.number to the mp20 cache on first run.

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

# ---- data ----
DATA_ROOT="${DATA_ROOT:-$CRYSFINITY_ROOT/data/mp20}"
DATASET_NAME="${DATASET_NAME:-mp20}"
NMAX="${NMAX:-20}"

# ---- conditioning ----
COND_PROP="${COND_PROP:-spacegroup}"
BASE_CKPT="${BASE_CKPT:-}"
if [[ -z "$BASE_CKPT" ]]; then
  if [[ -f "$CRYSFINITY_ROOT/dng_clean.pt" ]]; then
    BASE_CKPT="$CRYSFINITY_ROOT/dng_clean.pt"
  elif [[ -f "$CRYSFINITY_ROOT/dng.pt" ]]; then
    BASE_CKPT="$CRYSFINITY_ROOT/dng.pt"
  else
    if [[ -f "$PROJECT_ROOT/dng_clean.pt" ]]; then
      BASE_CKPT="$PROJECT_ROOT/dng_clean.pt"
    else
      BASE_CKPT="$PROJECT_ROOT/dng.pt"
    fi
  fi
fi
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
COND_DROPOUT="${COND_DROPOUT:-0.15}"

# ---- run identity ----
OUT_DIR="${OUT_DIR:-outputs/sg_cond}"
WANDB_NAME="${WANDB_NAME:-sg_cond}"
WANDB_PROJECT="${WANDB_PROJECT:-DNG_MP20_cond}"

# ---- optimisation / schedule ----
SEED="${SEED:-123}"
MAX_STEPS="${MAX_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-1000}"
EMA_DECAY="${EMA_DECAY:-0.9999}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# ---- logging / checkpointing / validation ----
LOG_EVERY="${LOG_EVERY:-50}"
VAL_EVERY="${VAL_EVERY:-2000}"
VAL_BATCHES="${VAL_BATCHES:-20}"
CKPT_EVERY="${CKPT_EVERY:-5000}"
# In-training DNG sampling/thermo OFF by default (evaluate via the sweep script).
SAMPLE_FREQUENCY="${SAMPLE_FREQUENCY:-0}"

# ---- architecture block: pinned to dng_clean.pt model_args (M0 recipe) ----
TYPE_ENCODING="${TYPE_ENCODING:-subatomic_tokenizer_pca_16}"
D_MODEL="${D_MODEL:-512}"
N_HEADS="${N_HEADS:-16}"
N_LAYERS="${N_LAYERS:-14}"
DROPOUT="${DROPOUT:-0.0}"
ATTN_DROPOUT="${ATTN_DROPOUT:-0.0}"
COORD_EMBED_MODE="${COORD_EMBED_MODE:-fourier}"
COORD_N_FREQS="${COORD_N_FREQS:-32}"
COORD_HEAD_MODE="${COORD_HEAD_MODE:-direct}"
COORD_LOSS_MODE="${COORD_LOSS_MODE:-frac_mse}"
EDGE_BIAS_HIDDEN_DIM="${EDGE_BIAS_HIDDEN_DIM:-256}"
EDGE_BIAS_N_FREQS="${EDGE_BIAS_N_FREQS:-12}"
EDGE_BIAS_N_RBF="${EDGE_BIAS_N_RBF:-32}"
PBC_RADIUS="${PBC_RADIUS:-1}"
LATTICE_EMBED_MODE="${LATTICE_EMBED_MODE:-mlp}"
LATTICE_REPR="${LATTICE_REPR:-ltri}"
LOSS_WEIGHTS="${LOSS_WEIGHTS:-1 50 5}"
EDM_P_MEAN="${EDM_P_MEAN:--1.2}"
EDM_P_STD="${EDM_P_STD:-1.2}"
SIGMA_DATA_TYPE="${SIGMA_DATA_TYPE:-0.3}"
SIGMA_DATA_COORD="${SIGMA_DATA_COORD:-0.3}"
SIGMA_DATA_LATTICE="${SIGMA_DATA_LATTICE:-0.3}"
SIGMA_MIN="${SIGMA_MIN:-0.002}"
SIGMA_MAX="${SIGMA_MAX:-80.0}"
RHO="${RHO:-7.0}"
S_CHURN="${S_CHURN:-60.0}"
S_MIN="${S_MIN:-0.0}"
S_MAX="${S_MAX:-999.0}"
S_NOISE="${S_NOISE:-1.003}"
AA_FRAC_MAX_SCALE="${AA_FRAC_MAX_SCALE:-0.0}"
AA_RHO_TYPES="${AA_RHO_TYPES:-0.0}"
AA_RHO_COORDS="${AA_RHO_COORDS:-10.0}"
AA_RHO_LATTICE="${AA_RHO_LATTICE:-10.0}"

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "[sg-cond] base checkpoint not found: $BASE_CKPT" >&2
  exit 2
fi

WANDB_ARGS=(--wandb_name "$WANDB_NAME" --wandb_project "$WANDB_PROJECT")
if [[ "${NO_WANDB:-0}" == "1" ]]; then
  WANDB_ARGS=(--no_wandb)
fi

echo "[sg-cond] project_root=$PROJECT_ROOT crysfinity_root=$CRYSFINITY_ROOT data_root=$DATA_ROOT"
echo "[sg-cond] cond_prop=$COND_PROP base_ckpt=$BASE_CKPT lora_rank=$LORA_RANK alpha=$LORA_ALPHA dropout=$COND_DROPOUT"
echo "[sg-cond] output_dir=$OUT_DIR max_steps=$MAX_STEPS batch_size=$BATCH_SIZE seed=$SEED"

srun uv run python src/train_crystalite.py \
  --dataset_name "$DATASET_NAME" \
  --data_root "$DATA_ROOT" \
  --nmax "$NMAX" \
  --bf16 \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --lr_warmup_steps "$LR_WARMUP_STEPS" \
  --ema_decay "$EMA_DECAY" \
  --max_steps "$MAX_STEPS" \
  --cond_prop "$COND_PROP" \
  --base_ckpt "$BASE_CKPT" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --cond_dropout "$COND_DROPOUT" \
  --type_encoding "$TYPE_ENCODING" \
  --d_model "$D_MODEL" \
  --n_heads "$N_HEADS" \
  --n_layers "$N_LAYERS" \
  --dropout "$DROPOUT" \
  --attn_dropout "$ATTN_DROPOUT" \
  --coord_embed_mode "$COORD_EMBED_MODE" \
  --coord_n_freqs "$COORD_N_FREQS" \
  --coord_head_mode "$COORD_HEAD_MODE" \
  --coord_loss_mode "$COORD_LOSS_MODE" \
  --use_edge_bias \
  --edge_bias_hidden_dim "$EDGE_BIAS_HIDDEN_DIM" \
  --edge_bias_n_freqs "$EDGE_BIAS_N_FREQS" \
  --edge_bias_n_rbf "$EDGE_BIAS_N_RBF" \
  --pbc_radius "$PBC_RADIUS" \
  --use_noise_gate \
  --lattice_embed_mode "$LATTICE_EMBED_MODE" \
  --lattice_repr "$LATTICE_REPR" \
  --loss_weights $LOSS_WEIGHTS \
  --edm_P_mean "$EDM_P_MEAN" \
  --edm_P_std "$EDM_P_STD" \
  --sigma_data_type "$SIGMA_DATA_TYPE" \
  --sigma_data_coord "$SIGMA_DATA_COORD" \
  --sigma_data_lattice "$SIGMA_DATA_LATTICE" \
  --sigma_min "$SIGMA_MIN" \
  --sigma_max "$SIGMA_MAX" \
  --rho "$RHO" \
  --S_churn "$S_CHURN" \
  --S_min "$S_MIN" \
  --S_max "$S_MAX" \
  --S_noise "$S_NOISE" \
  --aa_frac_max_scale "$AA_FRAC_MAX_SCALE" \
  --aa_rho_types "$AA_RHO_TYPES" \
  --aa_rho_coords "$AA_RHO_COORDS" \
  --aa_rho_lattice "$AA_RHO_LATTICE" \
  --sample_frequency "$SAMPLE_FREQUENCY" \
  --log_every "$LOG_EVERY" \
  --val_every "$VAL_EVERY" \
  --val_batches "$VAL_BATCHES" \
  --ckpt_every "$CKPT_EVERY" \
  --best_ckpt \
  --output_dir "$OUT_DIR" \
  "${WANDB_ARGS[@]}"

echo "[sg-cond] done -> $OUT_DIR/checkpoints (best.pt / step_*.pt)"
echo "[sg-cond] next: sbatch --export=ALL,CHECKPOINT=$OUT_DIR/checkpoints/best.pt scripts/sg_guidance_sweep.slurm"
