# Reproduction Status

Updated: 2026-07-12
Target paper-level Dice: `0.97`

## Summary

| Experiment | Mean Dice | Mean IoU | Vertebrae Dice | Canal Dice | IVD Dice | Best Epoch | Train Val Dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t2_space_reproduction_4cls090_cap1000_20260711 | 0.8836 | 0.8154 | 0.9182 | 0.7859 | 0.8451 | 32 | 0.8872 |

## Decision

Run `all_4cls090_cap1000` next. The current recorded results do not yet include the combined T1/T2/T2_SPACE condition, so the reproduction baseline is incomplete.

## Notes

- `Mean Dice` comes from `evaluate.py` class-wise validation metrics.
- `Train Val Dice` comes from the Keras training CSV and is secondary for paper comparison.
- Missing all-sequence results mean the reproduction baseline should still be treated as incomplete.
