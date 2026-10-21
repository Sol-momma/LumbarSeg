"""Layered config: experiment JSON (+extends) <- profile JSON <- env vars <- --set overrides."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO / "configs" / "experiments"
PROFILES = REPO / "configs" / "profiles"
ENV_OVERRIDES = {
    "LUMBARSEG_DATA_ROOT": "data_root",
    "LUMBARSEG_OUTPUT_ROOT": "output_root",
    "LUMBARSEG_RESULTS": "results_path",
}


class ConfigError(ValueError):
    pass


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_experiment(name_or_path: str) -> dict:
    path = Path(name_or_path)
    if not path.suffix:
        path = EXPERIMENTS / f"{name_or_path}.json"
    if not path.is_file():
        names = ", ".join(sorted(p.stem for p in EXPERIMENTS.glob("*.json")))
        raise ConfigError(f"Experiment config not found: {path}. Next: pass --config one of [{names}] or a JSON path.")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    parent = cfg.pop("extends", None)
    return _merge(_load_experiment(parent), cfg) if parent else cfg


def detect_profile() -> str:
    if os.environ.get("LUMBARSEG_PROFILE"):
        return os.environ["LUMBARSEG_PROFILE"]
    if Path("/content").is_dir() and os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    try:
        if "microsoft" in Path("/proc/version").read_text().lower():
            return "lab"
    except OSError:
        pass
    return "box"


def _coerce(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def load(experiment: str = "default", profile: str | None = None, sets: list[str] | None = None) -> dict:
    cfg = _load_experiment(experiment)
    profile = profile or detect_profile()
    ppath = PROFILES / f"{profile}.json"
    if not ppath.is_file():
        names = ", ".join(sorted(p.stem for p in PROFILES.glob("*.json")))
        raise ConfigError(f"Profile not found: {profile}. Next: use --profile one of [{names}] or set LUMBARSEG_PROFILE.")
    prof = json.loads(ppath.read_text(encoding="utf-8"))
    prof.pop("_doc", None)
    cfg["profile"] = {"name": profile, **prof}
    cfg["profile"].setdefault("results_path", str(REPO / "experiments" / "results.jsonl"))
    for env, key in ENV_OVERRIDES.items():
        if os.environ.get(env):
            cfg["profile"][key] = os.environ[env]
    for item in sets or []:
        if "=" not in item:
            raise ConfigError(f"Bad --set {item!r}. Next: use dotted KEY=VALUE, e.g. --set train.focal_gamma=0.4")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(raw)
    for key in ("data_root", "output_root", "results_path"):
        cfg["profile"][key] = str(Path(os.path.expanduser(cfg["profile"][key])))
    validate(cfg)
    return cfg


ALLOWED = {
    ("eval", "dice_average_unit"): {"slice", "case", "pixel"},
    ("eval", "surface_inf_policy"): {"propagate", "exclude", "max_distance"},
    ("data", "slice_cap_method"): {"linspace_sorted", "random"},
}


def validate(cfg: dict) -> None:
    for (sec, key), allowed in ALLOWED.items():
        val = cfg.get(sec, {}).get(key)
        if val not in allowed:
            raise ConfigError(f"{sec}.{key}={val!r} is invalid. Next: choose one of {sorted(allowed)} via --set {sec}.{key}=...")
    t, nc = cfg.get("train", {}), cfg.get("model", {}).get("num_classes", 4)
    if t.get("augment", "none") not in ("none", "strong"):
        raise ConfigError(f"train.augment={t.get('augment')!r} is invalid. Next: --set train.augment=none or strong")
    for key in ("class_weights", "dice_class_weights"):
        w = t.get(key)
        if w is not None and (not isinstance(w, list) or len(w) != nc or any(not isinstance(x, (int, float)) or x < 0 for x in w)):
            raise ConfigError(f"train.{key}={w!r} is invalid. Next: null or a list of {nc} non-negative numbers, e.g. --set 'train.{key}=[1,1,3,1]'")
    if not isinstance(cfg["eval"].get("include_background"), bool):
        raise ConfigError("eval.include_background must be true/false. Next: --set eval.include_background=false")


def public(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}
