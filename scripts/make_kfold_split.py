"""Write the patient-level k-fold assignment (patient -> fold) for the prepared slices to a JSON file.

The output lists SPIDER patient IDs, so write it OUTSIDE the repository (it is data-derived and not committed).
usage: python scripts/make_kfold_split.py --config paper_effective --out /tmp/kfold.json [--set data.kfold_seed=0]
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="paper_effective")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from lumbarseg_cli import config
    from lumbarseg_cli.cli import _split, kfold_assign, patient_id, paths
    from spine_baseline.preprocessing import get_series_id
    cfg = config.load(a.config, sets=a.set)
    data_root, out, _ = paths(cfg)
    tr, te, _ = _split(cfg, out / "processed", data_root, None)
    d = cfg["data"]
    fa = kfold_assign([patient_id(get_series_id(f)) for f in tr + te], int(d.get("n_folds", 5)), int(d.get("kfold_seed", 0)))
    out_p = Path(a.out).resolve()
    if Path(__file__).resolve().parents[1] in out_p.parents:
        sys.exit("refusing to write patient IDs inside the repository")
    out_p.write_text(json.dumps({"n_folds": int(d.get("n_folds", 5)), "kfold_seed": int(d.get("kfold_seed", 0)),
                                 "fold_of_patient": fa}, indent=1))
    print(f"{len(fa)} patients -> {out_p}")


if __name__ == "__main__":
    main()
