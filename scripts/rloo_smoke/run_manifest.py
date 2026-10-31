"""scripts/rloo_smoke/run_manifest.py

Resume guard. On a fresh start write run_manifest.json (raw values + hash of everything that affects the
optimisation). On --resume recompute and refuse on any mismatch, printing the diff. Also cross-checks max_steps
against the checkpoint's trainer_state.json, because changing max_steps on resume silently changes the LR schedule.

Fields that cannot change training are excluded (EXCLUDE). Use allow=("some.dotted.key",) for a deliberate override.
"""
import hashlib
import json
from pathlib import Path

NAME = "run_manifest.json"
EXCLUDE = {"output_root", "hub_repo", "report_to", "logging_steps", "save_total_limit"}


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def build_manifest(cfg, stats, extra=None):
    """cfg: parsed yaml. stats: StdStats. extra: anything else that must match (e.g. prompt-pool file sha256)."""
    rc = {k: v for k, v in cfg["run_config"].items() if k != "stats_path"}
    body = {
        "model_name": cfg["model_name"],
        "dataset_size": cfg["dataset_size"],
        "ensemble_mode": cfg["ensemble_mode"],
        "run_config": rc,
        "stats": {"r_center": stats.r_center, "r_scale": stats.r_scale,
                  "ensemble_id": stats.ensemble_id, "noise_condition": stats.noise_condition},
        "stats_file_sha256": _file_sha(cfg["run_config"].get("stats_path")),
        "lora": cfg["lora"],
        "rloo_config": {k: v for k, v in cfg["rloo_config"].items() if k not in EXCLUDE},
        "extra": extra or {},
    }
    blob = json.dumps(body, sort_keys=True, default=str)
    return {"sha256": hashlib.sha256(blob.encode()).hexdigest(), "body": body}


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def diff(old, new, allow=()):
    a, b = _flat(old), _flat(new)
    rows = []
    for k in sorted(set(a) | set(b)):
        if a.get(k, "<missing>") != b.get(k, "<missing>") and not any(k.startswith(p) for p in allow):
            rows.append(f"  {k}: checkpoint={a.get(k, '<missing>')!r}  now={b.get(k, '<missing>')!r}")
    return rows


def write_manifest(out_dir, manifest):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))


def check_resume(out_dir, manifest, ckpt=None, allow=()):
    """Raise SystemExit unless the manifest on disk matches. Call AFTER restore_from_hub, so a restored VM is checked."""
    p = Path(out_dir) / NAME
    if not p.exists():
        raise SystemExit(f"--resume: {p} missing, cannot verify the config. Restore it from the Hub or start fresh.")
    old = json.loads(p.read_text())
    rows = diff(old["body"], manifest["body"], allow)
    if ckpt:
        ts = Path(ckpt) / "trainer_state.json"
        if ts.exists():
            st = json.loads(ts.read_text())
            ms, done = st.get("max_steps"), st.get("global_step")
            now = manifest["body"]["rloo_config"].get("max_steps")
            if done is not None and now is not None and int(done) >= int(now):
                rows.append(f"  checkpoint is at step {done} >= max_steps {now}: run already finished, "
                            f"resume would extend it (never do that)")
            if ms is not None and now is not None and int(ms) != int(now):
                rows.append(f"  trainer_state.max_steps: checkpoint={ms} now={now}  (changes the LR schedule)")
    if rows:
        raise SystemExit("--resume refused, config differs from the run that wrote the checkpoint:\n" + "\n".join(rows))
