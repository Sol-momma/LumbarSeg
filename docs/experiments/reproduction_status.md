# Reproduction Status

Updated: 2026-08-05
Target foreground macro Dice: `0.9700`
Class floors: IVDs `0.9688`, Vertebrae `0.9712`, Spinal Canal `0.9671`

## Summary

| Experiment | Foreground Macro Dice | 4-Class Mean Dice | Mean IoU | Vertebrae Dice | Canal Dice | IVD Dice | Score Target | T2 SPACE Scope | Best Epoch | Train Val Dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| t2_space_reproduction_4cls090_cap1000_20260711 | 0.8497 | 0.8836 | 0.8154 | 0.9182 | 0.7859 | 0.8451 | MISS | UNVERIFIED | 32 | 0.8872 |

## Decision

Best recorded foreground macro Dice is 0.8497, below the target 0.9700, or at least one class is below its paper floor. Run the next controlled T2 SPACE candidate in the goal campaign.

## Notes

- `Foreground Macro Dice` averages IVDs, Vertebrae, and Spinal Canal only.
- `4-Class Mean Dice` is the historical evaluator row and includes Background; it is not the success oracle.
- `Score Target PASS` requires the foreground macro target and all three paper class floors.
- Campaign success additionally requires `T2 SPACE Scope VERIFIED` from `run_config.tsv`.
- `Train Val Dice` comes from the Keras training CSV and is secondary for paper comparison.
- Mixed-sequence results are diagnostic and must not replace the T2 SPACE success definition.
