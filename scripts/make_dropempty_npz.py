"""scripts/make_dropempty_npz.py

Write a copy of a joined (R, gold, empty, split, group) npz with empty responses removed from the candidate pool
(src.analysis_utils.drop_empty), so g1_g3_plots.py and g2_f1_calibration.py can be run on it unchanged.

python scripts/make_dropempty_npz.py --npz data/pilot_noise0_joined.npz --out data/pilot_noise0_joined_dropempty.npz [--min_frac 0.5]

Keeps R, gold, empty (all False), split, group. Drops prompt_ids/response_ids (ragged after the drop). Writes
<out>.drop_info.json (N before/after, removed prompts, discarded non-empty samples). Refuses test-split/test-named input.
"""
import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import analysis_utils as U  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_frac", type=float, default=0.5)
    a = ap.parse_args(argv)
    for p in (a.npz, a.out):
        if "test" in pathlib.Path(p).name.lower():
            raise SystemExit(f"refusing test-named path: {p}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = U.load_npz(a.npz)
    o = U.drop_empty(d, a.min_frac)
    kw = dict(R=o["R"], gold=o["gold"], empty=o["empty"], split=np.array(o["split"]))
    if o["group"] is not None:
        kw["group"] = o["group"]
    np.savez(a.out, **kw)
    pathlib.Path(a.out + ".drop_info.json").write_text(json.dumps(o["drop_info"], indent=1))
    print(json.dumps(o["drop_info"], indent=1))


if __name__ == "__main__":
    main()
