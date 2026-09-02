#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATASETS="${DATASETS:-ks50 vggsound}"
JSON_KS50="${JSON_KS50:-${ROOT}/data/json/ks50}"
JSON_VGG="${JSON_VGG:-${ROOT}/data/json/vgg}"
CHECKPOINT_KS50="${CHECKPOINT_KS50:-${ROOT}/checkpoints/cav_mae_ks50.pth}"
CHECKPOINT_VGG="${CHECKPOINT_VGG:-${ROOT}/checkpoints/vgg_65.5.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/firm_all21}"

cd "${ROOT}"

run_group() {
  local dataset="$1"
  local modality="$2"
  local json_root label_csv checkpoint corruption output
  if [[ "${dataset}" == "ks50" ]]; then
    json_root="${JSON_KS50}"
    label_csv="${ROOT}/configs/labels/class_labels_indices_ks50.csv"
    checkpoint="${CHECKPOINT_KS50}"
  else
    json_root="${JSON_VGG}"
    label_csv="${ROOT}/configs/labels/class_labels_indices_vgg.csv"
    checkpoint="${CHECKPOINT_VGG}"
  fi
  corruption="$([[ "${modality}" == "none" ]] && echo clean || echo all)"
  output="${OUTPUT_ROOT}/${dataset}_${modality}"
  if [[ -s "${output}/result.csv" ]]; then
    echo "[skip] ${output}/result.csv"
    return
  fi
  "${PYTHON}" run_firm.py \
    --dataset "${dataset}" \
    --json-root "${json_root}" \
    --label-csv "${label_csv}" \
    --checkpoint "${checkpoint}" \
    --gpu "${GPU}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --corruption-modality "${modality}" \
    --corruption "${corruption}" \
    --severity 5 \
    --output-dir "${output}"
}

for dataset in ${DATASETS}; do
  for modality in none video audio; do
    echo "===== ${dataset}/${modality} ====="
    run_group "${dataset}" "${modality}"
  done
done

"${PYTHON}" tools/summarize_results.py \
  --root "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/summary.csv"

