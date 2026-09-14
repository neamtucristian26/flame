#!/bin/bash

set -u
cd "$(dirname "$0")/.."
PY="${TS_BIN:-python}"
EPOCHS=100
GPUS=(0 1 2 3 4 5)

train_job() {
  local s=$1 f=$2 gpu=$3
  local sfx="_allheads$([ "$s" -eq 42 ] && echo '' || echo "_s${s}")"
  local out="artifacts/results/baseline_baseline_sep_fold${f}${sfx}.json"
  if [ -f "$out" ]; then
    echo "[$(date '+%F %T')] skip (exists) seed ${s} fold ${f}"; return
  fi
  echo "[$(date '+%F %T')] seed ${s} fold ${f} -> GPU ${gpu}"
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" scripts/baseline.py \
    --fold "${f}" --epochs "${EPOCHS}" --no-layer-warmstart \
    --seed "${s}" --run-suffix "${sfx}" \
    > "artifacts/logs/baseline_allheads_fold${f}_s${s}.log" 2>&1
  echo "[$(date '+%F %T')] seed ${s} fold ${f} done"
}

jS=(); jF=()
for s in 42 43 44; do
  for f in 0 1 2 3 4 5 6 7 8 9; do
    jS+=("$s"); jF+=("$f")
  done
done
n=${#jS[@]}
echo "[$(date '+%F %T')] [baseline-allheads] ${n} jobs @ ${EPOCHS} epochs"
for ((i=0; i<n; i+=6)); do
  pids=()
  for ((k=0; k<6 && i+k<n; k++)); do
    idx=$((i+k))
    train_job "${jS[$idx]}" "${jF[$idx]}" "${GPUS[$k]}" &
    pids+=($!)
  done
  wait "${pids[@]}"
done
echo "[$(date '+%F %T')] [baseline-allheads] training done -- aggregating"

"${PY}" - <<'PYEOF'
import json, numpy as np
from pathlib import Path

R = Path("artifacts/results")
HEADS = ["pair", "model", "arch_family", "acoustic_model", "vocoder",
         "speaker_type", "lang_family"]
METS  = ["acc", "macro_acc", "macro_f1"]
SEED_SFX = {42: "_allheads", 43: "_allheads_s43", 44: "_allheads_s44"}

per_seed = {h: {mt: {} for mt in METS} for h in HEADS}
ok = True
for s, sfx in SEED_SFX.items():
    vals = {h: {mt: [] for mt in METS} for h in HEADS}
    for f in range(10):
        p = R / f"baseline_baseline_sep_fold{f}{sfx}.json"
        if not p.exists():
            print(f"  MISSING seed {s} fold {f}"); ok = False; continue
        m = json.load(open(p))["metrics"]
        for h in HEADS:
            for mt in METS:
                vals[h][mt].append(m[h][mt])
    for h in HEADS:
        for mt in METS:
            per_seed[h][mt][s] = float(np.mean(vals[h][mt]))

agg = {}
for h in HEADS:
    agg[h] = {}
    for mt in METS:
        a = np.array([per_seed[h][mt][s] for s in SEED_SFX])
        agg[h][mt] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1))}

out = {"seeds": list(SEED_SFX), "heads": HEADS, "per_seed": per_seed, "agg": agg}
json.dump(out, open(R / "baseline_allheads_summary.json", "w"), indent=2)

print("\n=== BASELINE 7-head {42,43,44} SEEN (mean +/- sd over seeds) ===")
print(f"{'head':15}{'acc':>16}{'macro_acc':>16}{'macro_f1':>16}")
for h in HEADS:
    print(f"{h:15}" + "".join(
        f"{agg[h][mt]['mean']:.3f}±{agg[h][mt]['sd']:.3f}".rjust(16)
        for mt in METS))
print(f"\nWrote {R/'baseline_allheads_summary.json'}")
PYEOF

echo "[$(date '+%F %T')] [baseline-allheads] all done."
