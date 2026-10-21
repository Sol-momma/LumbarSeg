#!/usr/bin/env bash
# End-to-end harness proof in an isolated scratch dir. Usage: verify.sh [--download]
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"; cd "$REPO"
PY="${PY:-$REPO/.venv/bin/python}"; [ -x "$PY" ] || PY=python3
TS=$(date +%Y%m%d-%H%M%S); SCR="/tmp/lumbarseg-verify-$TS"; EVIDENCE="$SCR/evidence"; mkdir -p "$EVIDENCE"
export LUMBARSEG_OUTPUT_ROOT="$SCR/runs" LUMBARSEG_RESULTS="$SCR/results.jsonl"
fail() { echo "FAIL $*"; echo "evidence: $EVIDENCE"; exit 1; }
ok() { echo "PASS $*"; }
git rev-parse HEAD > "$EVIDENCE/commit.txt"
"$PY" -m lumbarseg_cli doctor --json > "$EVIDENCE/doctor.json"; dc=$?
"$PY" - "$EVIDENCE/doctor.json" <<'P' || fail "doctor JSON unreadable"
import json,sys; d=json.load(open(sys.argv[1])); [print(f"  {'ok ' if c['ok'] else '-- '}{c['check']}: {c['detail']}") for c in d['checks']]
P
"$PY" -m lumbarseg_cli prepare-data --download --dry-run --json > "$EVIDENCE/dryrun.json" || fail "download dry-run"
ok "download dry-run: $("$PY" -c "import json;d=json.load(open('$EVIDENCE/dryrun.json'))['download_plan'];print(d['files'],'files',round(d['bytes_to_download']/1e9,2),'GB')")"
before=$(find "$SCR/runs" -type f 2>/dev/null | wc -l)
"$PY" -m lumbarseg_cli cleanup all --config smoke --dry-run --json > "$EVIDENCE/cleanup-dry.json" || fail "cleanup dry-run"
[ "$(find "$SCR/runs" -type f 2>/dev/null | wc -l)" = "$before" ] && ok "dry-runs wrote nothing" || fail "dry-run wrote files"
"$PY" -m lumbarseg_cli smoke ${1:-} --json > "$EVIDENCE/smoke.json" 2> "$EVIDENCE/smoke.stderr.log" || fail "smoke (see smoke.stderr.log)"
"$PY" - "$EVIDENCE/smoke.json" "$LUMBARSEG_RESULTS" <<'P' || fail "smoke result checks"
import json,sys
s=json.load(open(sys.argv[1])); assert s["ok"], s
m=s["metrics"]["foreground_macro_dice"]; assert 0<=m<=1, m
lines=open(sys.argv[2]).read().splitlines(); assert len(lines)==1, len(lines)
r=json.loads(lines[0]); assert {"commit","config","split","metrics"}<=r.keys()
assert r["config"]["eval"]["dice_average_unit"]=="slice" and r["split"]["sha256"]
print(f"PASS smoke foreground_macro_dice={m:.4f} (pipeline only) record has commit/config/split/metrics")
P
ls "$SCR/runs/smoke/run/checkpoints/best_model.keras" >/dev/null || fail "no checkpoint"
cp "$LUMBARSEG_RESULTS" "$EVIDENCE/"
"$PY" -m lumbarseg_cli score-csv docs/experiments/t2_space_reproduction_4cls090_cap1000_20260711/validation_metrics.csv --json > "$EVIDENCE/score-csv.json"
"$PY" -c "import json,sys;sys.exit(round(json.load(open('$EVIDENCE/score-csv.json'))['foreground_macro_dice'],4)!=0.8497)" "$EVIDENCE/score-csv.json" && ok "score-csv recorded run = 0.8497" || fail "score-csv"
"$PY" -m lumbarseg_cli cleanup all --config smoke --json > "$EVIDENCE/cleanup.json" || fail cleanup
[ ! -d "$SCR/runs/smoke/run" ] && [ -f "$EVIDENCE/smoke.json" ] && ok "cleanup removed runs, evidence kept" || fail "cleanup"
echo "doctor exit=$dc (3 = a required check failed, e.g. no GPU on a GPU profile)"
echo "evidence: $EVIDENCE"
