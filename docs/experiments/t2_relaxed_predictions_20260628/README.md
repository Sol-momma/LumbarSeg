# T2 Relaxed Prediction Examples - 2026-06-28

Qualitative prediction examples from the T2 relaxed baseline.

Each PNG contains:

- Input MRI slice
- Ground truth mask
- Predicted mask
- Prediction overlay

The corresponding per-slice Dice scores are stored in `prediction_summary.csv`.

Model:

- `outputs/t2_relaxed/checkpoints/best_model.keras`
- Mean Dice: 0.889731
- Filtered slices: 1790
