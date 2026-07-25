#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --job-name=crystalite_cases
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x.%A.out
#SBATCH --error=%x.%A.err

# Reviewer-facing candidate case studies for generated Crystalite CIFs.
#
# This is CPU-only: it reads already-relaxed CIFs/manifest metadata, computes
# symmetry + oxidation plausibility, optionally searches MP20 references, and
# optionally extracts decomposition products from a phase-diagram pickle.
#
# Submit example:
#   sbatch --export=ALL,\
# PROJECT_ROOT=$HOME/crystalite,\
# CIF_DIR=$HOME/crysfinity/data/export/crystalite_m1_lossfix_s150/cifs,\
# REFERENCE_DATA_ROOT=$HOME/crysfinity/data/mp20,\
# PHASE_DIAGRAM=$HOME/crysfinity/mp_02072023/2023-02-07-ppd-mp.pkl,\
# OUT_DIR=outputs/candidate_case_studies \
#     scripts/run_candidate_case_studies.slurm
#
# If the server paths differ, override CIF_DIR, MANIFEST, REFERENCE_DATA_ROOT,
# PHASE_DIAGRAM, and OUT_DIR in --export.

set -euo pipefail

module purge
module load 2024

export PATH="$HOME/.local/bin:$PATH"

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/crystalite}"
cd "$PROJECT_ROOT"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/candidate_case_studies}"
mkdir -p "$LOG_DIR"

export UV_CACHE_DIR="${TMPDIR:-/tmp/$USER/uv-cache-${SLURM_JOB_ID:-manual}}"
mkdir -p "$UV_CACHE_DIR"

# ---- inputs ----
CIF_DIR="${CIF_DIR:-$HOME/crysfinity/data/export/crystalite_m1_lossfix_s150/cifs}"
MANIFEST="${MANIFEST:-$CIF_DIR/manifest.json}"
REFERENCE_DATA_ROOT="${REFERENCE_DATA_ROOT:-$HOME/crysfinity/data/mp20}"
PHASE_DIAGRAM="${PHASE_DIAGRAM:-$HOME/crysfinity/mp_02072023/2023-02-07-ppd-mp.pkl}"
OUT_DIR="${OUT_DIR:-outputs/candidate_case_studies}"

# ---- selection / analysis controls ----
TOP_K="${TOP_K:-5}"
MIN_E_ABOVE_HULL="${MIN_E_ABOVE_HULL:-0.0}"
MAX_E_ABOVE_HULL="${MAX_E_ABOVE_HULL:-0.1}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-500}"
NMAX="${NMAX:-20}"
REFERENCE_SPLITS="${REFERENCE_SPLITS:-train}"
MAX_REFERENCE_CANDIDATES="${MAX_REFERENCE_CANDIDATES:-5000}"
SYMPREC="${SYMPREC:-0.1}"
STOL="${STOL:-0.5}"
LTOL="${LTOL:-0.3}"
ANGLE_TOL="${ANGLE_TOL:-10}"

# Full rebuttal run defaults to requiring the server-only assets. Set either to
# 0 to produce a partial report instead of failing.
REQUIRE_REFERENCE="${REQUIRE_REFERENCE:-1}"
REQUIRE_PHASE_DIAGRAM="${REQUIRE_PHASE_DIAGRAM:-1}"
REQUIRE_SMACT_VALID="${REQUIRE_SMACT_VALID:-1}"

if [[ ! -d "$CIF_DIR" ]]; then
  echo "[case-studies] CIF_DIR not found: $CIF_DIR" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "[case-studies] MANIFEST not found: $MANIFEST" >&2
  exit 2
fi
if [[ "$REQUIRE_REFERENCE" == "1" && ! -d "$REFERENCE_DATA_ROOT" ]]; then
  echo "[case-studies] REFERENCE_DATA_ROOT not found: $REFERENCE_DATA_ROOT" >&2
  echo "[case-studies] set REQUIRE_REFERENCE=0 to run a partial report." >&2
  exit 2
fi
if [[ "$REQUIRE_PHASE_DIAGRAM" == "1" && ! -f "$PHASE_DIAGRAM" ]]; then
  echo "[case-studies] PHASE_DIAGRAM not found: $PHASE_DIAGRAM" >&2
  echo "[case-studies] set REQUIRE_PHASE_DIAGRAM=0 to run a partial report." >&2
  exit 2
fi

ARGS=(
  --cif_dir "$CIF_DIR"
  --manifest "$MANIFEST"
  --out_dir "$OUT_DIR"
  --top_k "$TOP_K"
  --min_e_above_hull "$MIN_E_ABOVE_HULL"
  --max_e_above_hull "$MAX_E_ABOVE_HULL"
  --candidate_pool_size "$CANDIDATE_POOL_SIZE"
  --nmax "$NMAX"
  --max_reference_candidates "$MAX_REFERENCE_CANDIDATES"
  --symprec "$SYMPREC"
  --stol "$STOL"
  --ltol "$LTOL"
  --angle_tol "$ANGLE_TOL"
)

if [[ "$REQUIRE_SMACT_VALID" == "1" ]]; then
  ARGS+=(--require_smact_valid)
fi

if [[ -d "$REFERENCE_DATA_ROOT" ]]; then
  # shellcheck disable=SC2206
  REFERENCE_SPLIT_ARGS=($REFERENCE_SPLITS)
  ARGS+=(--reference_data_root "$REFERENCE_DATA_ROOT" --reference_splits "${REFERENCE_SPLIT_ARGS[@]}")
else
  echo "[case-studies] reference search disabled; missing: $REFERENCE_DATA_ROOT"
fi

if [[ -f "$PHASE_DIAGRAM" ]]; then
  ARGS+=(--phase_diagram "$PHASE_DIAGRAM")
else
  echo "[case-studies] decomposition disabled; missing: $PHASE_DIAGRAM"
fi

echo "[case-studies] project_root=$PROJECT_ROOT"
echo "[case-studies] cif_dir=$CIF_DIR"
echo "[case-studies] manifest=$MANIFEST"
echo "[case-studies] reference_data_root=$REFERENCE_DATA_ROOT splits=$REFERENCE_SPLITS"
echo "[case-studies] phase_diagram=$PHASE_DIAGRAM"
echo "[case-studies] out_dir=$OUT_DIR top_k=$TOP_K hull_window=[$MIN_E_ABOVE_HULL,$MAX_E_ABOVE_HULL]"

srun uv run python scripts/candidate_case_studies.py "${ARGS[@]}"

echo "[case-studies] done. Outputs under: $OUT_DIR"
