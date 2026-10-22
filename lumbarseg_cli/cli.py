"""lumbarseg — one CLI for data prep, training, evaluation, environment checks and cleanup.

Every command accepts --json (machine-readable result on stdout, logs on stderr),
--profile {box,colab,lab} (or LUMBARSEG_PROFILE) and --config {default,paper,smoke|path}.
Exit codes: 0 ok, 1 runtime failure, 2 usage/config error, 3 environment not ready.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from lumbarseg_cli import config as cfgmod
from lumbarseg_cli.metrics import json_safe

REPO = cfgmod.REPO
OVERVIEW = "SPIDER Lumbar Spine Segmentation Overview.csv"


class CliError(Exception):
    def __init__(self, message: str, next_step: str, code: int = 1):
        super().__init__(message)
        self.next_step, self.code = next_step, code


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def emit(args, payload: dict, human: str | None = None):
    payload = json_safe(payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(human if human is not None else json.dumps(payload, ensure_ascii=False, indent=2))


def load_cfg(args) -> dict:
    try:
        return cfgmod.load(args.config, args.profile, args.set)
    except cfgmod.ConfigError as exc:
        msg = str(exc)
        raise CliError(msg.split(" Next: ")[0], msg.split(" Next: ")[-1], 2) from exc


def paths(cfg):
    p = cfg["profile"]
    out = Path(p["output_root"]) / cfg["name"]
    return Path(p["data_root"]), out, Path(p["results_path"])


def git_info() -> dict:
    def run(*c):
        try:
            return subprocess.run(["git", *c], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD") or "unknown", "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def t2_space_series(data_root: Path | None, sequences: str | None) -> set[str] | None:
    import pandas as pd
    csv = (data_root / OVERVIEW) if data_root and (data_root / OVERVIEW).is_file() else REPO / "data" / OVERVIEW
    names = pd.read_csv(csv)["new_file_name"].astype(str)
    if not sequences:
        return None
    from spine_baseline.preprocessing import classify_sequence, parse_sequences
    wanted = parse_sequences(sequences)
    return {n for n in names if classify_sequence(n) in wanted}


# ---------------------------------------------------------------- prepare-data
def cmd_prepare_data(args):
    cfg = load_cfg(args)
    data_root, out, _ = paths(cfg)
    d = cfg["data"]
    series = None if args.all_sequences else t2_space_series(data_root, d["sequences"])
    result = {"command": "prepare-data", "data_root": str(data_root), "processed_root": str(out / "processed"),
              "sequences": None if args.all_sequences else d["sequences"]}
    if args.download:
        from lumbarseg_cli import zenodo
        try:
            plan = zenodo.plan(series)
        except Exception as exc:
            raise CliError(f"Could not read Zenodo record: {exc}", "Check network access to zenodo.org, then retry. Data can also be placed manually (see README).") from exc
        free = shutil.disk_usage(data_root if data_root.exists() else data_root.parent if data_root.parent.exists() else Path.home()).free
        result["download_plan"] = {"files": plan["n_images"] + plan["n_masks"], "bytes_on_disk": plan["bytes"],
                                   "bytes_to_download": plan["download_bytes"], "free_bytes": free,
                                   "source": plan["record"], "license": plan["license"]}
        if plan["bytes"] > free:
            raise CliError(f"Need {plan['bytes']/1e9:.2f} GB but only {free/1e9:.2f} GB free at {data_root}",
                           "Free disk space, or narrow --set data.sequences=T2_SPACE, or set LUMBARSEG_DATA_ROOT to a bigger disk.", 3)
        if args.dry_run:
            result["dry_run"] = True
            return emit(args, result, f"[dry-run] would download {result['download_plan']['files']} files, "
                        f"{plan['download_bytes']/1e9:.2f} GB ({plan['bytes']/1e9:.2f} GB on disk) to {data_root}; free {free/1e9:.1f} GB. Nothing written.")
        result["download"] = zenodo.fetch(series, data_root, log=log)
    if not (data_root / "images").is_dir() or not (data_root / "masks").is_dir():
        raise CliError(f"No SPIDER images/ and masks/ under {data_root}",
                       "Run `python -m lumbarseg_cli prepare-data --download --dry-run` to see the size, then without --dry-run; or set LUMBARSEG_DATA_ROOT.", 3)
    if not (data_root / OVERVIEW).is_file():
        shutil.copy(REPO / "data" / OVERVIEW, data_root / OVERVIEW)
    if args.dry_run:
        n = len(list((data_root / "images").glob("*.mha")))
        result.update(dry_run=True, mha_found=n)
        return emit(args, result, f"[dry-run] would preprocess {n} volumes from {data_root} into {out/'processed'} (force={args.force}). Nothing written.")
    from spine_baseline.preprocessing import extract_slices, filter_slices
    proc = out / "processed"
    stats = extract_slices(data_root, proc, d["target_height"], d["target_width"], d["sequences"], force=args.force)
    kept, fstats = filter_slices(proc, d["min_classes"], d["imbalance_threshold"], d["max_slices_per_sequence"] or None,
                                 d["slice_cap_method"], d["slice_cap_seed"])
    if stats["errors"]:
        log("extraction errors:", stats["errors"][:5])
    result.update(extraction={k: v for k, v in stats.items() if k != "axes"}, filtering={k: v for k, v in fstats.items() if k != "corrupt_file_errors"})
    if not kept:
        raise CliError("Filtering kept 0 slices", "Relax --set data.imbalance_threshold=0.90 or check that masks match images.")
    emit(args, result, f"prepared {fstats['kept']} slices (from {fstats['before_filtering']}) in {proc}")


# ---------------------------------------------------------------- train
def _split(cfg, proc: Path, data_root: Path, smoke: dict | None):
    from spine_baseline.preprocessing import split_train_val
    filt = proc / "filtered_files.txt"
    if not filt.is_file():
        raise CliError(f"{filt} missing", f"Run `python -m lumbarseg_cli prepare-data --config {cfg['name']}` first.", 3)
    kept = [l for l in filt.read_text().splitlines() if l]
    train, val, unmatched = split_train_val(data_root, kept)
    if smoke:
        # Smoke: take evenly spaced slices so several series are touched.
        pick = lambda xs, n: [xs[i] for i in sorted({int(j) for j in __import__("numpy").linspace(0, len(xs) - 1, n)})] if xs else []
        train, val = pick(train, smoke["train_slices"]), pick(val, smoke["val_slices"])
    return train, val, unmatched


def split_with_test(cfg, proc: Path, data_root: Path, smoke: dict | None = None):
    """(train, early-stopping val, test, unmatched).

    The Overview-CSV validation cases are the fixed test set. With data.inner_val_cases=N>0, N training
    cases (picked by data.inner_val_seed, independent of train.seed) become the early-stopping set, so
    the test cases never steer model selection. N=0 keeps the old behaviour (val == test)."""
    import numpy as np
    from spine_baseline.preprocessing import get_series_id
    train, test, unmatched = _split(cfg, proc, data_root, smoke)
    n = int(cfg["data"].get("inner_val_cases", 0) or 0)
    if n <= 0:
        return train, test, test, unmatched
    from spine_baseline.preprocessing import classify_sequence
    by_seq: dict[str, list[str]] = {}
    for c in sorted({get_series_id(f) for f in train}):
        by_seq.setdefault(classify_sequence(c), []).append(c)
    inner = set()
    for seq in sorted(by_seq):  # N cases per sequence; with one sequence this is the original single draw
        cases = by_seq[seq]
        if n >= len(cases):
            raise CliError(f"data.inner_val_cases={n} >= {len(cases)} training cases of {seq}", "Use a smaller --set data.inner_val_cases=4", 2)
        rng = np.random.default_rng(int(cfg["data"].get("inner_val_seed", 0)))
        inner |= set(rng.choice(cases, n, replace=False).tolist())
    return [f for f in train if get_series_id(f) not in inner], [f for f in train if get_series_id(f) in inner], test, unmatched


def cmd_train(args):
    if args.smoke and args.config == "default":
        args.config = "smoke"
    cfg = load_cfg(args)
    data_root, out, _ = paths(cfg)
    proc = out / "processed"
    train_files, val_files, test_files, unmatched = split_with_test(cfg, proc, data_root, cfg.get("smoke") if args.smoke else None)
    if not train_files or not val_files:
        raise CliError(f"Empty split: train={len(train_files)} val={len(val_files)}",
                       "Check data.sequences and that the Overview CSV subset column covers the prepared series.")
    run_dir = out / "run"
    result = {"command": "train", "smoke": args.smoke, "run_dir": str(run_dir), "train_slices": len(train_files), "val_slices": len(val_files),
              "test_slices": len(test_files)}
    if args.dry_run:
        result["dry_run"] = True
        return emit(args, result, f"[dry-run] would train {cfg['train']['epochs']} epoch(s) on {len(train_files)} slices; outputs -> {run_dir}")
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from spine_baseline.dataset import create_dataset
    from spine_baseline.losses import combined_loss
    from spine_baseline.metrics import dice_coefficient, mean_iou
    from spine_baseline.model import build_modified_unet
    t, d, m = cfg["train"], cfg["data"], cfg["model"]
    np.random.seed(t["seed"]); tf.random.set_seed(t["seed"])
    ck = run_dir / "checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    for name, files in (("train_files.txt", train_files), ("validation_files.txt", val_files), ("test_files.txt", test_files),
                        ("unmatched_files.txt", unmatched)):
        (run_dir / name).write_text("\n".join(files) + "\n")
    (run_dir / "config.json").write_text(json.dumps(cfgmod.public(cfg), indent=2))
    aug = t.get("augment", "none")
    mk = lambda f, s: create_dataset(f, proc, d["target_height"], d["target_width"], m["num_classes"], t["batch_size"], shuffle=s,
                                     augment=aug if s else "none", seed=t["seed"],
                                     augment_workers=int(t.get("augment_workers", 1)))
    model = build_modified_unet(input_shape=(d["target_height"], d["target_width"], m["input_channels"]), num_classes=m["num_classes"],
                                dropout_rate=m["dropout_rate"], leaky_relu_alpha=m["leaky_relu_alpha"])
    model.compile(optimizer=keras.optimizers.Adam(t["learning_rate"]), loss=combined_loss(t["focal_weight"], t["focal_gamma"],
                                                                                     t.get("class_weights"), t.get("dice_class_weights")),
                  metrics=["accuracy", mean_iou(m["num_classes"]), dice_coefficient(m["num_classes"])])
    mon = t["monitor"]
    cbs = [keras.callbacks.ModelCheckpoint(str(ck / "best_model.keras"), monitor=mon, mode="max", save_best_only=True),
           keras.callbacks.EarlyStopping(monitor=mon, mode="max", patience=t["patience"], restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor=mon, mode="max", factor=0.5, patience=max(1, t["patience"] // 2), min_lr=1e-7),
           keras.callbacks.CSVLogger(str(ck / "training_log.csv"))]
    t0 = time.time()
    hist = model.fit(mk(train_files, True), validation_data=mk(val_files, False), epochs=t["epochs"], callbacks=cbs, verbose=2 if not args.json else 0)
    model.save(str(ck / "final_model.keras"))
    result.update(seconds=round(time.time() - t0, 1), epochs_run=len(hist.history["loss"]),
                  best_model=str(ck / "best_model.keras"), last_val_mean_iou=float(hist.history[mon][-1]))
    emit(args, result, f"trained {result['epochs_run']} epoch(s) in {result['seconds']}s -> {result['best_model']}")


# ---------------------------------------------------------------- evaluate
def cmd_evaluate(args):
    cfg = load_cfg(args)
    data_root, out, results_path = paths(cfg)
    proc = Path(args.processed_root) if args.processed_root else out / "processed"
    pred_dir = Path(args.pred_dir) if getattr(args, "pred_dir", None) else None
    model_path = pred_dir if pred_dir else (Path(args.model) if args.model else out / "run" / "checkpoints" / "best_model.keras")
    default_list = out / "run" / "test_files.txt"
    if not default_list.is_file():
        default_list = out / "run" / "validation_files.txt"
    file_list = Path(args.file_list) if args.file_list else default_list
    for pth, hint in ((model_path, "Train first (`train --smoke`), pass --model PATH.keras, or --pred_dir with <slice>.npz label maps"),
                      (file_list, "Pass --file_list with the frozen validation cohort (one .npz name per line)."),
                      (proc, "Run prepare-data first or pass --processed_root.")):
        if not pth.exists():
            raise CliError(f"Not found: {pth}", hint, 3)
    files = [l for l in file_list.read_text().splitlines() if l]
    if getattr(args, "sequence", None):
        from spine_baseline.preprocessing import classify_sequence, get_series_id as _sid
        files = [f for f in files if classify_sequence(_sid(f)) == args.sequence]
        if not files:
            raise CliError(f"No slices of {args.sequence} in {file_list}", "Check --sequence (T1, T2, T2_SPACE).", 2)
    if args.limit:
        files = files[: args.limit]
    import numpy as np
    import tensorflow as tf
    from spine_baseline.dataset import load_sample, load_slice_spacing
    from spine_baseline.preprocessing import get_series_id
    from lumbarseg_cli.metrics import aggregate, slice_metrics
    e, nc = cfg["eval"], cfg["model"]["num_classes"]
    model = None if pred_dir else tf.keras.models.load_model(str(model_path), compile=False)
    per = []
    for i, f in enumerate(files):
        img, oh = load_sample(f, proc, nc)
        sp = load_slice_spacing(f, proc)
        true = np.argmax(oh, -1)
        if pred_dir:
            pf = pred_dir / f
            if not pf.is_file():
                raise CliError(f"Prediction missing: {pf}", "Every slice in --file_list needs <pred_dir>/<name>.npz with key 'pred'.", 3)
            pred = np.load(pf)["pred"].astype(np.int64)
            if pred.shape != true.shape:
                raise CliError(f"{pf} shape {pred.shape} != mask {true.shape}", "Write predictions at the processed resolution.", 2)
        else:
            pred = np.argmax(model.predict(img[None], verbose=0)[0], -1)
        per.append({"series": get_series_id(f), "spacing": tuple(map(float, sp)), "shape": true.shape,
                    "classes": slice_metrics(pred, true, sp, nc, e["smooth"], e["nsd_tolerance_mm"])})
        if (i + 1) % 50 == 0:
            log(f"evaluated {i+1}/{len(files)}")
    metrics = aggregate(per, e, nc)
    g = git_info()
    record = {"command": "evaluate", "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "experiment": args.tag or cfg["name"],
              "commit": g["commit"], "dirty": g["dirty"], "profile": cfg["profile"]["name"], "model": str(model_path),
              "config": cfgmod.public({k: v for k, v in cfg.items() if k != "profile"}),
              "split": {"validation_list": str(file_list), "n_slices": len(files), "n_cases": metrics["n_cases"],
                        "sha256": sha256_lines(files), "limit": args.limit,
                        "sequence": getattr(args, "sequence", None)},
              "metrics": metrics}
    if args.no_record:
        record["recorded"] = False
    else:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
        record["recorded"] = str(results_path)
    c = metrics["classes"]
    human = (f"foreground_macro_dice={metrics['foreground_macro_dice']:.4f}  "
             + "  ".join(f"{n}={c[n]['dice']:.4f}" for n in list(c)[1:])
             + f"  (unit={e['dice_average_unit']}, n_slices={len(files)})\nrecorded: {record['recorded']}")
    emit(args, record, human)


def cmd_score_csv(args):
    """Recompute headline numbers from a legacy validation_metrics.csv (no model needed)."""
    import csv
    rows = {r["class"]: r for r in csv.DictReader(open(args.csv, encoding="utf-8"))}
    fg = ["Vertebrae", "Spinal Canal", "IVDs"]
    missing = [n for n in fg if n not in rows]
    if missing:
        raise CliError(f"{args.csv} lacks classes {missing}", "Pass a CSV produced by evaluate.py (columns class,dice,...).", 2)
    res = {"command": "score-csv", "csv": args.csv, "foreground_macro_dice": sum(float(rows[n]["dice"]) for n in fg) / 3,
           "all_class_mean_dice": sum(float(r["dice"]) for k, r in rows.items() if k != "Mean") / sum(k != "Mean" for k in rows),
           "classes": {n: float(rows[n]["dice"]) for n in rows if n != "Mean"}}
    emit(args, res, f"foreground_macro_dice={res['foreground_macro_dice']:.4f} all_class_mean_dice={res['all_class_mean_dice']:.4f}")


# ---------------------------------------------------------------- doctor
def _run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return p.returncode, (p.stdout or p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)


def cmd_doctor(args):
    cfg = load_cfg(args)
    data_root, out, results_path = paths(cfg)
    checks = []

    def add(name, ok, detail, fix=None, required=True):
        checks.append({"check": name, "ok": ok, "required": required, "detail": detail, **({"next": fix} if fix and not ok else {})})

    wsl = False
    try:
        wsl = "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        pass
    add("platform", True, f"{platform.system()} {platform.release()} python {platform.python_version()} wsl={wsl} profile={cfg['profile']['name']}")
    import importlib.util as iu
    deps = ["tensorflow", "SimpleITK", "cv2", "numpy", "pandas", "scipy", "sklearn", "tqdm"]
    missing = [m for m in deps if iu.find_spec(m) is None]
    add("python deps", not missing, "all present" if not missing else f"missing: {missing}",
        "pip install -r requirements-baseline.txt (CPU box: tensorflow-cpu)")
    want_gpu = cfg["profile"].get("device") == "gpu"
    code, smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    add("nvidia-smi", code == 0, smi.splitlines()[0] if code == 0 and smi else (smi or "not found")[:200],
        "WSL2: install the NVIDIA Windows driver (not a Linux driver) and run `wsl --update`; /usr/lib/wsl/lib/nvidia-smi should exist." if wsl
        else "Install NVIDIA driver or use a CPU profile (--profile box).", required=want_gpu)
    if iu.find_spec("tensorflow") is not None:
        code, tfg = _run([sys.executable, "-c", "import os;os.environ['TF_CPP_MIN_LOG_LEVEL']='3';import tensorflow as tf;print(tf.__version__, len(tf.config.list_physical_devices('GPU')))"])
        ver, n = (tfg.split()[-2:] if code == 0 else ("?", "0"))
        add("tensorflow GPU", n != "0", f"tf {ver}, GPUs visible: {n}",
            "On Linux/WSL2: pip install 'tensorflow[and-cuda]'. Native Windows TF has no GPU after 2.10 — run inside WSL2.", required=want_gpu)
    if iu.find_spec("torch") is not None:
        code, tg = _run([sys.executable, "-c", "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"])
        add("torch.cuda", code == 0 and " True " in f" {tg} ", tg[-200:], "pip install torch with the CUDA wheel matching nvidia-smi's CUDA version.", required=False)
    else:
        add("torch.cuda", False, "torch not installed (training uses TensorFlow; torch is only a second GPU probe)", "Optional: pip install torch", required=False)
    imgs = len(list((data_root / "images").glob("*.mha"))) if (data_root / "images").is_dir() else 0
    msks = len(list((data_root / "masks").glob("*.mha"))) if (data_root / "masks").is_dir() else 0
    add("data_root", imgs > 0 and imgs == msks, f"{data_root}: images={imgs} masks={msks}",
        "Run `python -m lumbarseg_cli prepare-data --download --dry-run`, or set LUMBARSEG_DATA_ROOT / --profile.")
    if wsl and str(data_root).startswith("/mnt/"):
        add("data on Windows drive", True, "data_root is on /mnt/* (9P, slow I/O). Preprocessed output goes to output_root; keep that on the Linux side.", required=False)
    proc = out / "processed"
    nproc = len(list((proc / "masks").glob("*.npz"))) if (proc / "masks").is_dir() else 0
    add("processed slices", nproc > 0, f"{proc}: {nproc} npz", f"python -m lumbarseg_cli prepare-data --config {cfg['name']}", required=False)
    probe = out if out.exists() else Path(cfg["profile"]["output_root"]).parent
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    add("disk free", free > 5e9, f"{free/1e9:.1f} GB free at {probe}", "Free space or move output_root (LUMBARSEG_OUTPUT_ROOT).")
    add("results log", True, f"{results_path} ({sum(1 for _ in open(results_path)) if results_path.exists() else 0} records)", required=False)
    ok = all(c["ok"] for c in checks if c["required"])
    res = {"command": "doctor", "ok": ok, "profile": cfg["profile"]["name"], "config": cfg["name"], "checks": checks}
    human = "\n".join(f"[{'OK ' if c['ok'] else ('FAIL' if c['required'] else 'warn')}] {c['check']}: {c['detail']}" + (f"\n       next: {c['next']}" if 'next' in c else "") for c in checks)
    emit(args, res, human + f"\nready: {ok}")
    return 0 if ok else 3


# ---------------------------------------------------------------- cleanup
def cmd_cleanup(args):
    cfg = load_cfg(args)
    _, out, _ = paths(cfg)
    targets = {"run": out / "run", "processed": out / "processed"}
    chosen = [targets[k] for k in (args.what if args.what != ["all"] else list(targets))]
    existing = [p for p in chosen if p.exists()]
    sizes = {str(p): sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) for p in existing}
    root = Path(cfg["profile"]["output_root"]).resolve()
    for p in existing:
        if root not in p.resolve().parents:
            raise CliError(f"Refusing to delete {p}: outside output_root {root}", "Fix profile output_root.", 2)
    res = {"command": "cleanup", "dry_run": args.dry_run, "targets": sizes, "note": "results.jsonl and raw data_root are never touched"}
    if not args.dry_run:
        for p in existing:
            shutil.rmtree(p)
    verb = "would remove" if args.dry_run else "removed"
    emit(args, res, "\n".join(f"{verb} {p} ({s/1e6:.1f} MB)" for p, s in sizes.items()) or "nothing to remove")


# ---------------------------------------------------------------- smoke (one-liner)
def cmd_smoke(args):
    base = ["--profile", args.profile] if args.profile else []
    steps = [["prepare-data", "--config", "smoke", *(["--download"] if args.download else [])],
             ["train", "--smoke"], ["evaluate", "--config", "smoke", "--tag", "smoke"]]
    env = dict(os.environ)
    # Smoke records are pipeline checks, not experiments: keep them out of the committed log.
    env.setdefault("LUMBARSEG_RESULTS", str(REPO / "experiments" / "smoke.local.jsonl"))
    summary = []
    for s in steps:
        cmd = [sys.executable, "-m", "lumbarseg_cli", *s, *base, "--json"]
        log("$ python " + " ".join(cmd[1:]))
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
        sys.stderr.write(p.stderr[-3000:])
        if p.returncode != 0:
            raise CliError(f"step `{s[0]}` failed (exit {p.returncode})", f"Re-run `python -m lumbarseg_cli {' '.join(s)}` to see the full error; run `doctor` first.", p.returncode)
        summary.append(json.loads(p.stdout))
    final = summary[-1]
    emit(args, {"command": "smoke", "ok": True, "metrics": final["metrics"], "recorded": final["recorded"], "steps": [s["command"] for s in summary]},
         f"smoke OK  foreground_macro_dice={final['metrics']['foreground_macro_dice']:.4f} (pipeline check only, not accuracy)\nrecorded: {final['recorded']}")


EPILOG = """examples:
  python -m lumbarseg_cli doctor                         # GPU / data / deps check
  python -m lumbarseg_cli prepare-data --download --dry-run   # show Zenodo download size, write nothing
  python -m lumbarseg_cli smoke --download --json        # prepare -> tiny train -> evaluate -> metrics JSON
  python -m lumbarseg_cli train --config paper           # paper settings (gamma 0.4, batch 8, threshold 0.55)
  python -m lumbarseg_cli evaluate --model M.keras --file_list frozen_val.txt --processed_root DIR
  python -m lumbarseg_cli cleanup all --dry-run

profiles (--profile or LUMBARSEG_PROFILE; auto: colab / wsl->lab / box):
  box    /workspace/lumbarseg-data/DataSet, CPU
  colab  /content/drive/MyDrive/SPIDER/DataSet, GPU
  lab    /mnt/c/Users/ctlab/somomma/DataSet, WSL2 + RTX 3060 Ti
env overrides: LUMBARSEG_DATA_ROOT, LUMBARSEG_OUTPUT_ROOT, LUMBARSEG_RESULTS
exit codes: 0 ok, 1 failure, 2 usage/config error, 3 environment not ready"""


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print one JSON object on stdout (logs go to stderr)")
    common.add_argument("--profile", choices=None, default=None, help="machine profile: box | colab | lab (default: auto-detect / LUMBARSEG_PROFILE)")
    common.add_argument("--config", default="default", help="experiment config: default (reproduces 0.8497 settings) | paper | smoke | path/to.json")
    common.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value, e.g. --set train.focal_gamma=0.4 (repeatable)")
    p = argparse.ArgumentParser(prog="python -m lumbarseg_cli", description=__doc__, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")
    fmt = argparse.RawDescriptionHelpFormatter
    s = sub.add_parser("prepare-data", parents=[common], formatter_class=fmt, help="(optionally download) and preprocess SPIDER into 2D slices",
                       description="Download SPIDER from Zenodo (only the configured sequences, via HTTP range reads of images.zip) and/or\n"
                                   "extract + filter sagittal slices into <output_root>/<config>/processed.\nRaw data stays outside the repo; it is never committed.")
    s.add_argument("--download", action="store_true", help="fetch missing .mha files from Zenodo 10.5281/zenodo.10159290 (CC-BY-4.0)")
    s.add_argument("--all-sequences", action="store_true", help="with --download: fetch every sequence (~3.8 GB) instead of data.sequences")
    s.add_argument("--force", action="store_true", help="re-extract slices even if they exist (overwrites processed/)")
    s.add_argument("--dry-run", action="store_true", help="show download size / work to do, write nothing")
    s.set_defaults(fn=cmd_prepare_data)
    s = sub.add_parser("train", parents=[common], formatter_class=fmt, help="train the Modified U-Net",
                       description="Train with the selected config. --smoke uses configs/experiments/smoke.json (128x160, 8 train / 4 val slices, 1 epoch).\nWrites run/{train,validation,unmatched}_files.txt, config.json and checkpoints/.")
    s.add_argument("--smoke", action="store_true", help="tiny CPU run to prove the pipeline (not an accuracy number)")
    s.add_argument("--dry-run", action="store_true", help="show split sizes and outputs, do not train or overwrite checkpoints")
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("evaluate", parents=[common], formatter_class=fmt, help="evaluate a model and append to experiments/results.jsonl",
                       description="Counting rules come from config.eval (dice_average_unit, include_background, surface_inf_policy) and are\nrecorded with commit, config, split hash and metrics in results.jsonl. See README_HARNESS.md.")
    s.add_argument("--model", help="path to .keras (default: <output>/run/checkpoints/best_model.keras)")
    s.add_argument("--file_list", help="frozen evaluation cohort (default: <output>/run/test_files.txt, else validation_files.txt)")
    s.add_argument("--processed_root", help="directory with images/ and masks/ npz (default: <output>/processed)")
    s.add_argument("--sequence", choices=["T1", "T2", "T2_SPACE"], help="score only slices of this sequence (per-sequence test)")
    s.add_argument("--pred_dir", help="score saved label maps (<slice>.npz, key 'pred') instead of a Keras model, e.g. nnU-Net output")
    s.add_argument("--limit", type=int, help="evaluate only the first N slices (quick check; recorded in split.limit)")
    s.add_argument("--tag", help="experiment name stored in the record (default: config name)")
    s.add_argument("--no-record", action="store_true", help="do not append to results.jsonl")
    s.set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("score-csv", parents=[common], help="recompute headline Dice from a legacy validation_metrics.csv")
    s.add_argument("csv")
    s.set_defaults(fn=cmd_score_csv)
    s = sub.add_parser("doctor", parents=[common], formatter_class=fmt, help="check GPU (nvidia-smi, TF, torch.cuda), data location, deps, disk",
                       description="Read-only. Exit 0 when all required checks pass, 3 otherwise. GPU checks are required only for GPU profiles.")
    s.set_defaults(fn=cmd_doctor)
    s = sub.add_parser("cleanup", parents=[common], help="delete run/ and/or processed/ under output_root")
    s.add_argument("what", nargs="+", choices=["run", "processed", "all"])
    s.add_argument("--dry-run", action="store_true", help="list what would be deleted and sizes")
    s.set_defaults(fn=cmd_cleanup)
    s = sub.add_parser("smoke", parents=[common], help="one line: prepare-data -> train --smoke -> evaluate -> metrics JSON")
    s.add_argument("--download", action="store_true", help="fetch T2 SPACE data from Zenodo first if missing (~1.6 GB)")
    s.set_defaults(fn=cmd_smoke)
    return p


def main(argv=None):
    sys.path.insert(0, str(REPO))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    args = build_parser().parse_args(argv)
    try:
        rc = args.fn(args)
        return rc or 0
    except CliError as exc:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(exc), "next": exc.next_step}, ensure_ascii=False))
        log(f"error: {exc}\nnext: {exc.next_step}")
        return exc.code
