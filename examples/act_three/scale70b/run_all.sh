#!/usr/bin/env bash
# Run the full 70B scale test on ONE node.  Each stage resumes where it stopped,
# so the script can simply be re-run after a failure or pre-emption.
#
#   cd <repo root>
#   bash examples/act_three/scale70b/run_all.sh                   # all stages
#   STAGES="5 6 7 8" bash examples/act_three/scale70b/run_all.sh   # some stages
#
# Multi-node: run stages R and 0-4 once, then stage 5 on N nodes with
#   SHARD_INDEX=i NUM_SHARDS=N STAGES=5 bash .../run_all.sh
# then stages 6-8 once after all shards finish.
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
REF_MODEL="${REF_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
STAGES="${STAGES:-R 0 1 2 3 4 5 6 7 8}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
# Optional ablations; add e.g. "repe_const repe_only_s1 hybrid_s1"
EXTRA_CONDITIONS="${EXTRA_CONDITIONS:-repe_const}"
export ACT70B_DTYPE="${ACT70B_DTYPE:-float16}"
export ACT70B_QUANT="${ACT70B_QUANT:-none}"
export ACT70B_MAX_MEMORY="${ACT70B_MAX_MEMORY:-}"
export PYTHONUNBUFFERED=1

PKG=examples.act_three.scale70b
SHORT=$(python -c "import importlib.util,sys;s=importlib.util.spec_from_file_location('s','examples/act_three/scale70b/settings.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);print(m.short_name('$MODEL'))")
MDIR=examples/act_three/models/$SHORT
LOGDIR=examples/act_three/experiments/results/scale/$SHORT/logs
REFDIR=examples/act_three/scale70b/reference
mkdir -p "$MDIR" "$LOGDIR" "$REFDIR"

has() { [[ " $STAGES " == *" $1 "* ]]; }
run() { local name=$1; shift; echo "=== [$name] $*"; "$@" 2>&1 | tee -a "$LOGDIR/$name.log"; }

# Stage R: 8B reference (about 30 min on one GPU).  Uses the paper's 8B
# directions and reader from the repository.  Must run before stages 3-4:
#  - the norm probe on 8B gives the reference perturbation size (-> s_match at 70B);
#  - the strength rule is applied to 8B to check that it reproduces the paper's
#    choice (s = 1) before it is used to choose the 70B strength.
if has R; then
  if [[ ! -f "$REFDIR/norm_probe_llama3.json" ]]; then
    run sR_norm python -m $PKG.s03_norm_probe --model "$REF_MODEL"
  fi
  if [[ ! -f "$REFDIR/calibration_llama3.json" ]]; then
    run sR_cal python -m $PKG.s04_strength_calibration --model "$REF_MODEL"
    cp examples/act_three/experiments/results/scale/llama3/calibration.json "$REFDIR/calibration_llama3.json"
  fi
  python - "$REFDIR/calibration_llama3.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["chosen_strength"]
print(f"[R] 8B rule check: chose s = {s} (paper used s = 1)" + ("" if s == 1 else "  <-- differs; please report this"))
PY
fi

if has 0; then
  run s00 python -m $PKG.s00_preflight --model "$MODEL" --skip-reader-check
fi

if has 1; then
  if [[ ! -f "$MDIR/epa_directions.pkl" ]]; then
    run s01_extract python -m $PKG.run_cli examples.act_three.extract_directions --model "$MODEL"
  fi
  if [[ ! -f "$MDIR/epa_reading_tuning_v2_results.json" ]]; then
    run s01_tune python -m $PKG.run_cli examples.act_three.tune_reader --model "$MODEL" \
      --directions "$MDIR/epa_directions.pkl" \
      --train-data examples/act_three/models/llama3/epa_tuning_dataset_train.json \
      --test-data examples/act_three/models/llama3/epa_tuning_dataset_test.json \
      --output "$MDIR/epa_reading_tuning_v2_results.json"
  fi
  # Full preflight now that a reader exists (multi-GPU steering + reader equivalence)
  run s00_full python -m $PKG.s00_preflight --model "$MODEL"
fi

if has 2; then run s02 python -m $PKG.s02_targets --model "$MODEL"; fi
if has 3; then run s03 python -m $PKG.s03_norm_probe --model "$MODEL"; fi
if has 4; then run s04 python -m $PKG.s04_strength_calibration --model "$MODEL" --write-config; fi

if has 5; then
  run "s05_shard${SHARD_INDEX}" python -m $PKG.s05_main_steering --model "$MODEL" \
    --conditions unsteered repe_only pe_only hybrid $EXTRA_CONDITIONS \
    --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS"
fi

if has 6; then run s06 python -m $PKG.s06_coherence --model "$MODEL"; fi

if has 7; then
  # 70B outputs read by the 8B reader (independent measure) ...
  run s07_a python -m $PKG.s07_cross_read --reader-model "$REF_MODEL" --generator "$MODEL"
  # ... and the paper's 8B outputs read by the 70B reader.
  run s07_b python -m $PKG.s07_cross_read --reader-model "$MODEL" --generator "$REF_MODEL"
fi

if has 8; then
  run s08 python examples/act_three/scale70b/s08_merge_analyze.py --model "$MODEL"
  OUT=act_scale_results_${SHORT}_$(date +%Y%m%d).tar.gz
  tar czf "$OUT" \
    examples/act_three/experiments/results/scale \
    "$REFDIR" \
    "$MDIR/config.json" "$MDIR/epa_reading_tuning_v2_results.json" "$MDIR/contrastive_pairs.json" \
    "$MDIR/epa_directions.pkl" 2>/dev/null || true
  echo "Results packaged in $OUT: please send this file back."
fi
