"""src/chunked.py: 
resumable chunked processing for interruptible (free-tier Colab) runs.

Each chunk -> shard_XXXXX.jsonl written atomically; existing shards are skipped on rerun.
manifest.json pins (ids, chunk_size, meta): a mismatched resume exits instead of mixing runs.
"""

import hashlib, json, os
from pathlib import Path


def _sha(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def run_chunked(ids, fn, out_dir, chunk_size, meta=None, on_shard=None):
    """ids: ordered list[str]. fn(chunk_ids) -> list of JSON-serialisable rows, same length/order.
    on_shard(path): optional hook, e.g. HF Hub upload; failures print but never kill the run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man = {"ids_sha": _sha(ids), "n": len(ids), "chunk_size": chunk_size, "meta": meta or {}}
    mp = out / "manifest.json"

    if mp.exists():
        old = json.loads(mp.read_text())
        if old != man:
            raise SystemExit(f"{mp} differs from this run (ids/chunk_size/meta); use a new out_dir")
    else:
        mp.write_text(json.dumps(man, indent=2, sort_keys=True))

    for k, s in enumerate(range(0, len(ids), chunk_size)):
        shard = out / f"shard_{k:05d}.jsonl"
        if shard.exists():
            continue
        chunk = ids[s:s + chunk_size]
        rows = fn(chunk)
        assert len(rows) == len(chunk), "fn must return one row per id"
        tmp = shard.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        os.replace(tmp, shard)

        if on_shard:
            try:
                on_shard(shard)
            except Exception as e:  # noqa: BLE001
                print(f"!!! on_shard failed for {shard.name}: {e}")

    return read_all(out_dir, len(ids))


def read_all(out_dir, n_expected=None):
    rows = []

    for f in sorted(Path(out_dir).glob("shard_*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if n_expected is not None and len(rows) != n_expected:
        raise SystemExit(f"{out_dir}: {len(rows)} rows, expected {n_expected}: incomplete")

    return rows