"""Count sagittal slices per sequence (T1 / T2 / T2_SPACE / all) for the whole SPIDER set and how many
survive several readings of the paper's "class imbalance ratio > 55%" filter. CPU only, no training,
writes only counts (no image data).

Slices are produced exactly like spine_baseline.preprocessing.extract_slices: smallest axis = sagittal,
constant-intensity slices skipped, labels mapped to 4 classes, mask resized to 512x640 (nearest).
usage: python scripts/count_filter_all.py --data_root /workspace/SPIDER/DataSet --out counts.json [--md table.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# name -> (description, keep(fractions[4]) for slices that already contain all 4 classes)
DEFS = {
    "A": ("現実装: 前景3クラス内の最大クラス割合 ≤ 0.55", lambda c: c[1:].max() / c[1:].sum() <= 0.55),
    "A90": ("参考: 同じ定義で ≤ 0.90（実装の既定値）", lambda c: c[1:].max() / c[1:].sum() <= 0.90),
    "B": ("背景も含めた最大クラスの割合 ≤ 0.55", lambda c: c.max() <= 0.55),
    "C": ("前景各クラスの画像全体に対する割合の最大 ≤ 0.55", lambda c: c[1:].max() <= 0.55),
    "D": ("最小クラス / 最大クラス（4クラス）≥ 0.55", lambda c: c.min() / c.max() >= 0.55),
    "E": ("前景の最小 / 最大 ≥ 0.55", lambda c: c[1:].min() / c[1:].max() >= 0.55),
    "F": ("前景合計の割合 ≤ 0.55", lambda c: c[1:].sum() <= 0.55),
    "G": ("最大 / 最小（4クラス）を%で見て ≤ 55（=最大が最小の55倍以下）", lambda c: c.max() / c.min() <= 55),
}


def count_series(args):
    image_path, mask_path, h, w = args
    import cv2
    import SimpleITK as sitk
    from spine_baseline.preprocessing import classify_sequence, iter_sagittal_slices, map_labels
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.int16)
    res = {"sequence": classify_sequence(image_path.name), "series": image_path.stem, "total": 0, "four": 0,
           "kept": {k: 0 for k in DEFS}, "fractions": []}
    if image.shape != mask.shape:
        res["error"] = f"shape mismatch {image.shape} vs {mask.shape}"
        return res
    for _, _, img, msk in iter_sagittal_slices(image, mask):
        if img.max() == img.min():
            continue
        res["total"] += 1
        m = cv2.resize(map_labels(msk), (w, h), interpolation=cv2.INTER_NEAREST)
        c = np.bincount(m.ravel(), minlength=4)[:4] / m.size
        if (c > 0).sum() < 4:
            continue
        res["four"] += 1
        res["fractions"].append(c.tolist())
        for k, (_, keep) in DEFS.items():
            res["kept"][k] += bool(keep(c))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    imgs = sorted((a.data_root / "images").glob("*.mha"))
    jobs = [(p, a.data_root / "masks" / p.name, a.height, a.width) for p in imgs if (a.data_root / "masks" / p.name).is_file()]
    with ProcessPoolExecutor(a.workers) as ex:
        series = list(ex.map(count_series, jobs))
    groups = {"T1": [], "T2": [], "T2_SPACE": [], "ALL": []}
    for s in series:
        groups.setdefault(s["sequence"], []).append(s); groups["ALL"].append(s)
    summary = {}
    for g, ss in groups.items():
        fr = np.array([f for s in ss for f in s["fractions"]]) if any(s["fractions"] for s in ss) else np.zeros((0, 4))
        summary[g] = {"series": len(ss), "total_slices": sum(s["total"] for s in ss), "four_class_slices": sum(s["four"] for s in ss),
                      "kept": {k: sum(s["kept"][k] for s in ss) for k in DEFS},
                      "kept_cases": {k: sum(1 for s in ss if s["kept"][k]) for k in DEFS},
                      "mean_class_fraction": fr.mean(0).round(4).tolist() if len(fr) else None}
    errors = [f"{s['series']}: {s['error']}" for s in series if "error" in s]
    out = {"n_series": len(series), "definitions": {k: v[0] for k, v in DEFS.items()}, "summary": summary, "errors": errors}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    if a.md:
        order = ["T1", "T2", "T2_SPACE", "ALL"]
        lines = ["| 定義 | " + " | ".join(order) + " |", "|---|" + "---|" * len(order),
                 "| 系列数 | " + " | ".join(str(summary[g]["series"]) for g in order) + " |",
                 "| 全スライス | " + " | ".join(str(summary[g]["total_slices"]) for g in order) + " |",
                 "| 4クラスそろうスライス | " + " | ".join(str(summary[g]["four_class_slices"]) for g in order) + " |"]
        for k, (desc, _) in DEFS.items():
            lines.append(f"| {k}: {desc} | " + " | ".join(f"{summary[g]['kept'][k]}（{summary[g]['kept_cases'][k]}系列）" for g in order) + " |")
        a.md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"n_series": len(series), "errors": len(errors)}))


if __name__ == "__main__":
    main()
