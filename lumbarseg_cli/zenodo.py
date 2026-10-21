"""Fetch SPIDER (Zenodo 10.5281/zenodo.10159290) without downloading all of images.zip.

images.zip is ~3.7 GB. Zenodo serves HTTP Range requests, so we read the zip
central directory remotely and pull only the members we need (e.g. T2 SPACE).
Files land under the user's data_root only; nothing is written into the repo.
"""
from __future__ import annotations

import io
import os
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

RECORD_ID = "10159290"
API = f"https://zenodo.org/api/records/{RECORD_ID}"
OVERVIEW_NAME = "SPIDER Lumbar Spine Segmentation Overview.csv"


class HttpRangeFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP Range requests."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status != 206:
                raise OSError(f"Server ignored Range request (HTTP {resp.status}) for {self.url}")
            data = resp.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def record_files() -> dict[str, dict]:
    with urllib.request.urlopen(API, timeout=60) as resp:
        record = json.load(resp)
    return {f["key"]: {"size": f["size"], "url": f["links"]["self"]} for f in record["files"]}


def open_remote_zip(info: dict) -> zipfile.ZipFile:
    raw = HttpRangeFile(info["url"], info["size"])
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=8 * 1024 * 1024))


def plan(series: set[str] | None) -> dict:
    """Return what would be fetched (read-only; only touches zip directories)."""
    files = record_files()
    out = {"record": f"https://doi.org/10.5281/zenodo.{RECORD_ID}", "license": "CC-BY-4.0", "members": [], "bytes": 0}
    for key in ("images.zip", "masks.zip"):
        with open_remote_zip(files[key]) as zf:
            for m in zf.infolist():
                stem = Path(m.filename).name.removesuffix(".mha")
                if m.is_dir() or not m.filename.endswith(".mha"):
                    continue
                if series is not None and stem not in series:
                    continue
                out["members"].append({"zip": key, "name": m.filename, "compressed": m.compress_size, "size": m.file_size})
    out["bytes"] = sum(m["size"] for m in out["members"]) + files["overview.csv"]["size"]
    out["download_bytes"] = sum(m["compressed"] for m in out["members"]) + files["overview.csv"]["size"]
    out["n_images"] = sum(m["zip"] == "images.zip" for m in out["members"])
    out["n_masks"] = sum(m["zip"] == "masks.zip" for m in out["members"])
    return out


def fetch(series: set[str] | None, data_root: Path, log=print, workers: int = 8) -> dict:
    files = record_files()
    p = plan(series)
    by_zip: dict[str, list[str]] = {}
    for m in p["members"]:
        by_zip.setdefault(m["zip"], []).append(m["name"])
    from concurrent.futures import ThreadPoolExecutor

    def one(key: str, name: str) -> bool:
        sub = "images" if key == "images.zip" else "masks"
        dest = data_root / sub / Path(name).name
        with open_remote_zip(files[key]) as zf:  # one handle per worker
            if dest.exists() and dest.stat().st_size == zf.getinfo(name).file_size:
                return False
            tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")
            with zf.open(name) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            tmp.rename(dest)
        log(f"fetched {sub}/{dest.name}")
        return True

    jobs = [(k, n) for k, names in by_zip.items() for n in names]
    for k in by_zip:
        (data_root / ("images" if k == "images.zip" else "masks")).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = list(pool.map(lambda j: one(*j), jobs))
    written, skipped = sum(done), len(done) - sum(done)
    overview = data_root / OVERVIEW_NAME
    if not overview.exists():
        with urllib.request.urlopen(files["overview.csv"]["url"], timeout=120) as resp:
            overview.write_bytes(resp.read())
    return {"written": written, "skipped_existing": skipped, "data_root": str(data_root), **{k: p[k] for k in ("bytes", "n_images", "n_masks")}}
