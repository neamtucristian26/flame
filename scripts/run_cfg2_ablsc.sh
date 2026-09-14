#!/bin/bash

set -u
cd "$(dirname "$0")/.."
PY="${TS_BIN:-python}"
EPOCHS="${EPOCHS:-100}"
SEED="${1:-42}"
CONFIG="${CONFIG:-A0}"

LAMBDA="${LAMBDA:-0.298}"
GAMMA="${GAMMA:-0.401}"
TAU_A="${TAU_A:-0.045}"
TAU_CM="${TAU_CM:-0.02}"
TAU_T="${TAU_T:-0.045}"
K="${K:-32}"
W_VOC="0.0"
W_AC="0.0"
AUG_TEXT="${AUG_TEXT:-}"
ATTR_TMPL="${ATTR_TMPL:-}"
ATTR_TMPL_PROB="${ATTR_TMPL_PROB:-}"
MASK_TMPL_TEXT="${MASK_TMPL_TEXT:-}"
ALL_DESC="${ALL_DESC:-}"
CROSS_MODAL_LOSS="${CROSS_MODAL_LOSS:-}"
LEARN_TEMPS="${LEARN_TEMPS:-}"
SAVE_EVERY="${SAVE_EVERY:-}"

case "${CONFIG}" in
    A0) TAU_CM="${TAU_A}"; TAU_T="${TAU_A}"; K=32  ;;
    A1) TAU_CM=0.02;       TAU_T="${TAU_A}"; K=32  ;;
    A2) TAU_CM=0.03;       TAU_T="${TAU_A}"; K=32  ;;
    A3) TAU_CM="${TAU_A}"; TAU_T=0.07;       K=32  ;;
    A4) TAU_CM="${TAU_A}"; TAU_T=0.10;       K=32  ;;
    A5) TAU_CM="${TAU_A}"; TAU_T="${TAU_A}"; K=16  ;;
    A6) TAU_CM="${TAU_A}"; TAU_T="${TAU_A}"; K=8   ;;
    B0) TAU_CM=0.02;       TAU_T="${TAU_A}"; K=32  ;;
    B1) TAU_CM=0.02;       TAU_T="${TAU_A}"; K=32  ;;
    P_*|T_*|A_*|K_*|MP_*|F5_*) ;;
    *)  echo "Unknown CONFIG=${CONFIG}"; exit 1 ;;
esac

FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"

cfg_lower=$(echo "${CONFIG}" | tr '[:upper:]' '[:lower:]')
SUFFIX="${SUFFIX:-ablsc_${cfg_lower}}"

echo "[cfg2-ablsc] config=${CONFIG} seed=${SEED} τ_cm=${TAU_CM} τ_a=${TAU_A} τ_t=${TAU_T} K=${K} "\
"λ=${LAMBDA} γ=${GAMMA} w_voc=${W_VOC} w_ac=${W_AC} aug=${AUG_TEXT:-none} attr_tmpl=${ATTR_TMPL:-none} "\
"p_tmpl=${ATTR_TMPL_PROB:-uniform} mask_text=${MASK_TMPL_TEXT:-0} all_desc=${ALL_DESC:-0} epochs=${EPOCHS} folds='${FOLDS}'"

train_fold () {
    local f=$1 gpu=$2
    echo "[$(date '+%F %T')] seed ${SEED} fold ${f} -> GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=${gpu} "${PY}" scripts/train_zero_shot.py \
        --held-out-models 14 --epochs ${EPOCHS} --seed ${SEED} --fold ${f} \
        --loss-mode all_supcon \
        --supcon-weight ${LAMBDA} --text-supcon-weight ${GAMMA} \
        --supcon-temperature ${TAU_A} \
        --cross-modal-temperature ${TAU_CM} \
        --text-supcon-temperature ${TAU_T} \
        --vocoder-template-weight ${W_VOC} --acoustic-template-weight ${W_AC} \
        ${AUG_TEXT:+--augmented-text-path ${AUG_TEXT}} \
        ${ATTR_TMPL:+--attribute-template-path ${ATTR_TMPL}} \
        ${ATTR_TMPL_PROB:+--attr-tmpl-prob ${ATTR_TMPL_PROB}} \
        ${MASK_TMPL_TEXT:+--mask-tmpl-from-text-supcon} \
        ${ALL_DESC:+--all-desc-positives} \
        ${CROSS_MODAL_LOSS:+--cross-modal-loss ${CROSS_MODAL_LOSS}} \
        ${LEARN_TEMPS:+--learnable-supcon-temps} \
        ${SAVE_EVERY:+--save-every ${SAVE_EVERY}} \
        --scale-max 100 \
        --k-per-class ${K} --batch-size 256 \
        --labels-path data/labels_lang.json --arch-labels-path data/arch_labels.json \
        --no-layer-warmstart \
        ${SKIP_FINAL_EVAL:+--skip-final-eval} \
        --run-suffix "_${SUFFIX}_f${f}_s${SEED}" \
        > "artifacts/logs/cfg2_${SUFFIX}_fold${f}_s${SEED}.log" 2>&1
    echo "[$(date '+%F %T')] seed ${SEED} fold ${f} done"
}

GPU_BASE="${GPU_BASE:-0}"
GPU_COUNT="${GPU_COUNT:-6}"
pids=(); i=0
for f in ${FOLDS}; do
    gpu=$(( GPU_BASE + (i % GPU_COUNT) ))
    train_fold ${f} ${gpu} & pids+=($!)
    i=$((i+1))
    if [ $(( i % GPU_COUNT )) -eq 0 ]; then wait "${pids[@]}"; pids=(); fi
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
echo "[$(date '+%F %T')] CFG2 ABLSC ${CONFIG} (seed ${SEED}) DONE."
