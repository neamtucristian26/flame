#!/bin/bash

set -u
cd "$(dirname "$0")/.."
PY="${TS_BIN:-python}"
AUG="artifacts/embedding_cache/lang_passage_aug24ho4_e5-large-v2.npy"
TMPL25="artifacts/embedding_cache/lang_passage_attrtmpl25_e5-large-v2.npy"
CFG="F5_p10ho4"
SUFFIX="ablsc_f5_p10ho4"
export HELD_OUT_IDX=4
log() { echo "[$(date '+%F %T')] $*"; }

train_seed() {
  env CONFIG="$CFG" TAU_CM=0.020 TAU_T=0.07 MASK_TMPL_TEXT=1 ATTR_TMPL_PROB=0.10 \
      AUG_TEXT="$AUG" ATTR_TMPL="$TMPL25" HELD_OUT_IDX=4 \
      FOLDS="0 1 2 3 4 5 6 7 8 9" GPU_BASE="$2" GPU_COUNT="$3" \
      bash scripts/run_cfg2_ablsc.sh "$1" \
      > "artifacts/logs/cfg2_${SUFFIX}_s$1_master.log" 2>&1
}

log "STAGE 1a: train seeds 42 (GPUs 0-2) + 43 (GPUs 3-5)"
train_seed 42 0 3 & p42=$!
train_seed 43 3 3 & p43=$!
wait "$p42"; wait "$p43"
log "STAGE 1b: train seed 44 (GPUs 0-5)"
train_seed 44 0 6
log "STAGE 1 done"

log "STAGE 2: ensemble eval (3 seeds) + attribute probe"
CUDA_VISIBLE_DEVICES=0 HELD_OUT_IDX=4 "$PY" scripts/eval_sibling_ensemble.py \
  --ckpt-tmpl "best_lc_zs_e5-large-v2__ho14_fold{f}_${SUFFIX}_f{f}_s{s}.pt" \
  --out-tag f5_p10ho4 \
  > artifacts/logs/eval_ensemble_f5_p10ho4.log 2>&1
for s in 42 43 44; do
  CUDA_VISIBLE_DEVICES=0 HELD_OUT_IDX=4 "$PY" scripts/eval_attribute_retrieval.py \
    --ckpt-tmpl "best_lc_zs_e5-large-v2__ho14_fold{f}_${SUFFIX}_f{f}_s${s}.pt" \
    --seed "$s" --anchor-prefix query --out-suffix "_${SUFFIX}_s${s}_query" \
    > "artifacts/logs/eval_attr_${SUFFIX}_s${s}_query.log" 2>&1
done
log "STAGE 2 done"
log "F5_p10ho4 3-SEED COMPLETE."
