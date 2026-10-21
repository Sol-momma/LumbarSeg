#!/usr/bin/env bash
# (c) nnU-Net v2 2D on the default split. Runs inside a GPU pod from the repo root.
# env: OUT (run dir), LUMBARSEG_* already exported, NNUNET_TRAINER (default nnUNetTrainer_100epochs)
set -euo pipefail
TR="${NNUNET_TRAINER:-nnUNetTrainer_100epochs}"
pip install -q nnunetv2 "tensorflow-cpu==2.17.*"
export nnUNet_raw=/tmp/nnunet/raw nnUNet_preprocessed=/tmp/nnunet/pre nnUNet_results=$OUT/nnunet_results
export nnUNet_n_proc_DA=$(( $(nproc) > 10 ? 10 : $(nproc) ))
python -m lumbarseg_cli prepare-data --config default --json > $OUT/prepare.json
python scripts/nnunet2d.py export --config default --raw $nnUNet_raw --splits /tmp/splits_final.json --lists $OUT/run
nnUNetv2_plan_and_preprocess -d 501 -c 2d --verify_dataset_integrity -np 8
cp /tmp/splits_final.json $nnUNet_preprocessed/Dataset501_LumbarT2S/splits_final.json
t0=$(date +%s)
nnUNetv2_train 501 2d 0 -tr $TR
echo "nnunet_train_seconds=$(( $(date +%s) - t0 ))"
V=$(ls -d $nnUNet_results/Dataset501_LumbarT2S/${TR}__nnUNetPlans__2d/fold_0/validation)
python scripts/nnunet2d.py import --val_dir $V --file_list $OUT/run/validation_files.txt --pred_dir $OUT/nnunet_pred
for u in slice case pixel; do
  python -m lumbarseg_cli evaluate --config default --set eval.dice_average_unit=$u --pred_dir $OUT/nnunet_pred \
    --file_list $OUT/run/validation_files.txt --tag "nnunet2d|val|$u" --json > $OUT/eval_$u.json
done
