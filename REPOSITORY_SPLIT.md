# Repository Split Plan

LumbarSeg is split into two repositories, following the same separation as FastGS:

| Repository | Purpose |
| --- | --- |
| `Sol-momma/LumbarSeg` | Research code, training/evaluation CLI, dataset notes, reproducibility instructions |
| `lumbarseg.github.io` | Public project website built with Astro |

## Code Repository

Keep the main repository focused on reproducible experiments:

- `preprocess.py`, `train.py`, `evaluate.py`
- `spine_baseline/`
- `arguments/`
- `data/`
- `requirements-baseline.txt`
- research and workflow documentation

The code repository should link to the website, but should not carry the website build toolchain.

## Website Repository

Keep the website repository focused on static publication:

- `src/`
- `public/`
- `astro.config.mjs`
- `package.json`
- `package-lock.json`
- `tsconfig.json`

Use this repository for GitHub Pages. If publishing under the `Sol-momma` account root page, rename the remote repository to `Sol-momma.github.io`.
