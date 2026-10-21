"""nnU-Net v2 2D baseline on exactly the same slices and train/val split as the `default` config.

  export  : processed npz slices -> nnUNet_raw/Dataset501_LumbarT2S (one 2D case per slice) + splits_final.json
  import  : nnU-Net fold_0/validation/*.nii.gz -> <pred_dir>/<slice>.npz (key 'pred')
Then score with `python -m lumbarseg_cli evaluate --pred_dir <pred_dir> --file_list validation_files.txt`.
Spacing is written as 1x1 so nnU-Net trains at the processed resolution (no resampling), like the U-Net.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
DS = "Dataset501_LumbarT2S"


def case_id(name: str) -> str:
    return Path(name).stem


def export(a):
    import SimpleITK as sitk
    from lumbarseg_cli import config as cfgmod
    from lumbarseg_cli.cli import _split
    cfg = cfgmod.load(a.config, None, a.set)
    proc = Path(cfg["profile"]["output_root"]) / cfg["name"] / "processed"
    train, val, _ = _split(cfg, proc, Path(cfg["profile"]["data_root"]), None)
    out = Path(a.raw) / DS
    for sub in ("imagesTr", "labelsTr"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for f in train + val:
        img = np.load(proc / "images" / f)["image"].astype(np.float32)
        msk = np.load(proc / "masks" / f)["mask"].astype(np.uint8)
        for arr, path in ((img, out / "imagesTr" / f"{case_id(f)}_0000.nii.gz"), (msk, out / "labelsTr" / f"{case_id(f)}.nii.gz")):
            im = sitk.GetImageFromArray(arr[None]); im.SetSpacing((1.0, 1.0, 1.0)); sitk.WriteImage(im, str(path))
    (out / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "MRI"}, "labels": {"background": 0, "vertebrae": 1, "spinal_canal": 2, "ivd": 3},
        "numTraining": len(train) + len(val), "file_ending": ".nii.gz"}, indent=1))
    splits = [{"train": [case_id(f) for f in train], "val": [case_id(f) for f in val]}]
    Path(a.splits).write_text(json.dumps(splits))
    Path(a.lists).mkdir(parents=True, exist_ok=True)
    (Path(a.lists) / "train_files.txt").write_text("\n".join(train) + "\n")
    (Path(a.lists) / "validation_files.txt").write_text("\n".join(val) + "\n")
    print(json.dumps({"train": len(train), "val": len(val), "raw": str(out)}))


def import_preds(a):
    import SimpleITK as sitk
    out = Path(a.pred_dir); out.mkdir(parents=True, exist_ok=True)
    names = [l for l in Path(a.file_list).read_text().splitlines() if l]
    for f in names:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(Path(a.val_dir) / f"{case_id(f)}.nii.gz")))[0]
        np.savez_compressed(out / f, pred=arr.astype(np.uint8))
    print(json.dumps({"imported": len(names), "pred_dir": str(out)}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--config", default="default"); e.add_argument("--set", action="append", default=[])
    e.add_argument("--raw", required=True); e.add_argument("--splits", required=True); e.add_argument("--lists", required=True)
    e.set_defaults(fn=export)
    i = sub.add_parser("import"); i.add_argument("--val_dir", required=True); i.add_argument("--file_list", required=True)
    i.add_argument("--pred_dir", required=True); i.set_defaults(fn=import_preds)
    a = p.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
