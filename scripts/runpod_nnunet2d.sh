#!/usr/bin/env bash
# nnU-Net v2 2D on the harness split. Runs inside a GPU pod from the repo root.
# env: EVAL_SEQS (optional, per-sequence scoring), OUT (run dir), LUMBARSEG_* exported, NNUNET_TRAINER (default nnUNetTrainer_100epochs),
#      NNUNET_CONFIG (harness config for the split, default "default"), NNUNET_SETS (extra --set, space separated),
#      NNUNET_TAG (record tag, default nnunet2d)
set -euo pipefail
TR="${NNUNET_TRAINER:-nnUNetTrainer_100epochs}"; CFG="${NNUNET_CONFIG:-default}"; TAG="${NNUNET_TAG:-nnunet2d}"
SETS=(); for s in ${NNUNET_SETS:-}; do SETS+=(--set "$s"); done
pip install -q nnunetv2 "tensorflow-cpu==2.17.*" "numpy<2" "pandas==2.2.3"  # pandas 3 needs numpy>=2, which TF 2.17 cannot use
export nnUNet_raw=/tmp/nnunet/raw nnUNet_preprocessed=/tmp/nnunet/pre nnUNet_results=$OUT/nnunet_results
export nnUNet_n_proc_DA=$(( $(nproc) > 10 ? 10 : $(nproc) ))
rm -rf /tmp/nnunet
python -m lumbarseg_cli prepare-data --config "$CFG" "${SETS[@]}" --json > $OUT/prepare.json
python scripts/nnunet2d.py export --config "$CFG" "${SETS[@]}" --raw $nnUNet_raw --splits /tmp/splits_final.json --lists $OUT/run
nnUNetv2_plan_and_preprocess -d 501 -c 2d --verify_dataset_integrity -np 8
cp /tmp/splits_final.json $nnUNet_preprocessed/Dataset501_LumbarT2S/splits_final.json
t0=$(date +%s)
nnUNetv2_train 501 2d 0 -tr $TR
echo "nnunet_train_seconds=$(( $(date +%s) - t0 ))"
R=$nnUNet_results/Dataset501_LumbarT2S/${TR}__nnUNetPlans__2d
if [ -d $nnUNet_raw/Dataset501_LumbarT2S/imagesTs ]; then
  nnUNetv2_predict -i $nnUNet_raw/Dataset501_LumbarT2S/imagesTs -o $OUT/nnunet_test -d 501 -c 2d -f 0 -tr $TR
  V=$OUT/nnunet_test
else V=$R/fold_0/validation; fi
python scripts/nnunet2d.py import --val_dir $V --file_list $OUT/run/test_files.txt --pred_dir $OUT/nnunet_pred
# EVAL_SEQS="T1 T2 T2_SPACE" scores each sequence separately; empty = all test slices together
for q in ${EVAL_SEQS:-ALL}; do QA=(); [ "$q" != ALL ] && QA=(--sequence $q)
for u in slice case pixel; do
  python -m lumbarseg_cli evaluate --config "$CFG" "${SETS[@]}" --set eval.dice_average_unit=$u --pred_dir $OUT/nnunet_pred \
    --file_list $OUT/run/test_files.txt "${QA[@]}" --tag "$TAG|test|$q|$u" --json > $OUT/eval_${q}_$u.json
done; done
