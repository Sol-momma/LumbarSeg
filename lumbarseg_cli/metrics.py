"""Pure-numpy evaluation with the counting rules fixed by config (see README "Evaluation protocol")."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy import ndimage

CLASS_NAMES = ["Background", "Vertebrae", "Spinal Canal", "IVDs"]


def _surface(mask):
    if not mask.any():
        return np.zeros_like(mask, bool)
    return mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)


def surface_distances(pred, true, spacing):
    """Symmetric surface distances (mm). Empty/empty -> [0]; one side empty -> [inf] (same as spine_baseline.metrics)."""
    ps, ts = _surface(pred), _surface(true)
    if not ps.any() and not ts.any():
        return np.array([0.0])
    if not ps.any() or not ts.any():
        return np.array([np.inf])
    d_pt = ndimage.distance_transform_edt(~ts, sampling=spacing)[ps]
    d_tp = ndimage.distance_transform_edt(~ps, sampling=spacing)[ts]
    return np.concatenate([d_pt, d_tp])


def slice_metrics(pred2d, true2d, spacing, num_classes, smooth, nsd_tol):
    rows = []
    for c in range(num_classes):
        p, t = pred2d == c, true2d == c
        inter, ps, ts = int((p & t).sum()), int(p.sum()), int(t.sum())
        d = surface_distances(p, t, spacing)
        finite = np.isfinite(d).all()
        rows.append({
            "inter": inter, "pred": ps, "true": ts,
            "dice": (2 * inter + smooth) / (ps + ts + smooth),
            "iou": (inter + smooth) / (ps + ts - inter + smooth),
            "asd": float(d.mean()) if finite else math.inf,
            "hd": float(d.max()) if finite else math.inf,
            "hd95": float(np.percentile(d, 95)) if finite else math.inf,
            "nsd": float((d <= nsd_tol).mean()) if finite else 0.0,
        })
    return rows


def _agg_surface(values, policy, fallback):
    arr = np.asarray(values, float)
    n_inf = int(np.isinf(arr).sum())
    if policy == "propagate":
        val = float(arr.mean())
    elif policy == "exclude":
        val = float(arr[np.isfinite(arr)].mean()) if np.isfinite(arr).any() else math.inf
    else:  # max_distance
        val = float(np.where(np.isinf(arr), fallback, arr).mean())
    return val, n_inf


def aggregate(per_slice: list[dict], cfg_eval: dict, num_classes: int) -> dict:
    """per_slice items: {"series": str, "spacing": (r,c), "shape": (h,w), "classes": slice_metrics(...)}"""
    unit, smooth, policy = cfg_eval["dice_average_unit"], cfg_eval["smooth"], cfg_eval["surface_inf_policy"]
    out = {"classes": {}, "n_slices": len(per_slice), "n_cases": len({s["series"] for s in per_slice})}
    for c in range(num_classes):
        name = CLASS_NAMES[c]
        rows = [s["classes"][c] for s in per_slice]
        if unit == "slice":
            dice = float(np.mean([r["dice"] for r in rows]))
            iou = float(np.mean([r["iou"] for r in rows]))
        elif unit == "pixel":
            i, p, t = (sum(r[k] for r in rows) for k in ("inter", "pred", "true"))
            dice, iou = (2 * i + smooth) / (p + t + smooth), (i + smooth) / (p + t - i + smooth)
        else:  # case: pool pixels within a series, then average series
            by = defaultdict(lambda: [0, 0, 0])
            for s, r in zip(per_slice, rows):
                acc = by[s["series"]]
                acc[0] += r["inter"]; acc[1] += r["pred"]; acc[2] += r["true"]
            dice = float(np.mean([(2 * i + smooth) / (p + t + smooth) for i, p, t in by.values()]))
            iou = float(np.mean([(i + smooth) / (p + t - i + smooth) for i, p, t in by.values()]))
        fallback = [float(np.hypot(s["shape"][0] * s["spacing"][0], s["shape"][1] * s["spacing"][1])) for s in per_slice]
        entry = {"dice": dice, "iou": iou, "nsd": float(np.mean([r["nsd"] for r in rows]))}
        for key in ("asd", "hd", "hd95"):
            vals = [r[key] for r in rows]
            if policy == "max_distance":
                vals = [fb if math.isinf(v) else v for v, fb in zip(vals, fallback)]
                entry[key], entry[f"{key}_n_inf"] = float(np.mean(vals)), sum(math.isinf(r[key]) for r in rows)
            else:
                entry[key], entry[f"{key}_n_inf"] = _agg_surface(vals, policy, None)
        out["classes"][name] = entry
    fg = CLASS_NAMES[1:num_classes]
    used = CLASS_NAMES[:num_classes] if cfg_eval["include_background"] else fg
    out["foreground_macro_dice"] = float(np.mean([out["classes"][n]["dice"] for n in fg]))
    out["all_class_mean_dice"] = float(np.mean([out["classes"][n]["dice"] for n in CLASS_NAMES[:num_classes]]))
    out["mean_dice"] = float(np.mean([out["classes"][n]["dice"] for n in used]))
    out["mean_iou"] = float(np.mean([out["classes"][n]["iou"] for n in used]))
    return out


def json_safe(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return "inf" if obj > 0 else ("-inf" if obj < 0 else "nan")
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    return obj
