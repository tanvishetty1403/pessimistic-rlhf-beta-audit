"""scripts/proxy_gold_accuracy.py
Per-member / ensemble pairwise accuracy of the proxy RMs vs human labels and vs gold, joined BY pair_id.

  --gold  jsonl: pair_id, gold_chosen, gold_rejected (optional: group)   [gold_score.py --smoke format]
  --proxy NAME=path.npz (repeat): pair_ids (n,), chosen (n,K), rejected (n,K)   <- ASSUMED export format; adapt
  --diff A B : paired cluster-bootstrap CI of (ens acc vs gold, A) - (ens acc vs gold, B)

'chosen' = ORIGINAL human-chosen text (clean val pairs): acc vs human = P(s_chosen > s_rejected).
Gold ties are excluded from vs-gold numbers. Proxy ties count 0.5. CIs: bootstrap over `group` if present.
"""

import argparse, json, os, tempfile
import numpy as np


def load_gold(p):
    g = {}
    for l in open(p):
        if l.strip():
            d = json.loads(l)
            assert d["pair_id"] not in g, f"duplicate pair_id {d['pair_id']}"
            g[d["pair_id"]] = d
    return g


def boot(v, mask, inv, G, B, rng):
    sv = np.bincount(inv, np.where(mask, v, 0.0), G)
    cn = np.bincount(inv, mask.astype(float), G)
    idx = rng.integers(0, G, (B, G))
    bs = sv[idx].sum(1) / np.maximum(cn[idx].sum(1), 1)
    return sv.sum() / cn.sum(), np.percentile(bs, [2.5, 97.5])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--proxy", nargs="+", required=True)
    ap.add_argument("--diff", nargs=2, default=None)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(0)

    gold = load_gold(a.gold)
    prox = {}

    for spec in a.proxy:
        name, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=False)
        ids = [str(i) for i in z["pair_ids"]]
        assert len(set(ids)) == len(ids), f"{name}: duplicate pair_ids"
        prox[name] = dict(row={i: k for k, i in enumerate(ids)}, c=z["chosen"], r=z["rejected"])

    common = sorted(set(gold).intersection(*[set(v["row"]) for v in prox.values()]))
    assert common, "no pair_ids in common"
    print(f"pairs: gold={len(gold)} common={len(common)} (dropped {len(gold) - len(common)} from gold side)")
    gsgn = np.sign(np.array([gold[i]["gold_chosen"] - gold[i]["gold_rejected"] for i in common]))
    gmask = gsgn != 0
    groups = [str(gold[i].get("group", i)) for i in common]
    _, inv = np.unique(groups, return_inverse=True)
    G = int(inv.max()) + 1
    allm = np.ones(len(common), bool)

    m, h, _ = boot((gsgn > 0).astype(float), gmask, inv, G, a.B, rng), None, None
    print(f"gold-vs-human agreement (ties dropped, n={int(gmask.sum())}): {m[0]:.3f}  CI [{m[1][0]:.3f}, {m[1][1]:.3f}]")

    res, ens_gold = {}, {}
    for name, v in prox.items():
        rows = np.array([v["row"][i] for i in common])
        M = v["c"][rows] - v["r"][rows]                      # (n, K) margins
        cols = {f"member{k}": M[:, k] for k in range(M.shape[1])}
        cols["ens"] = M.mean(1)
        print(f"\n== {name} ==\n{'':9s} {'vs human':>24s}   {'vs gold':>24s}")

        for cn_, mm in cols.items():
            w_h = (1 + np.sign(mm)) / 2
            w_g = (1 + np.sign(mm) * gsgn) / 2
            ah, (hl, hh) = boot(w_h, allm, inv, G, a.B, rng)
            ag, (gl, gh) = boot(w_g, gmask, inv, G, a.B, rng)
            print(f"{cn_:9s} {ah:.3f} [{hl:.3f},{hh:.3f}]   {ag:.3f} [{gl:.3f},{gh:.3f}]")
            res[f"{name}/{cn_}"] = dict(vs_human=ah, vs_gold=ag)
            if cn_ == "ens":
                ens_gold[name] = w_g

    if a.diff:
        d = ens_gold[a.diff[0]] - ens_gold[a.diff[1]]
        dm, (dl, dh) = boot(d, gmask, inv, G, a.B, rng)
        print(f"\nens vs gold, {a.diff[0]} - {a.diff[1]}: {dm:+.3f} CI [{dl:+.3f},{dh:+.3f}]")
        res["diff_ens_vs_gold"] = dict(mean=dm, ci95=[dl, dh])

    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


def selftest():
    rng = np.random.default_rng(1)
    n, K = 600, 5
    ids = np.array([f"train-{i}" for i in range(n)])
    gc, gr = rng.normal(size=n), rng.normal(size=n)
    d = tempfile.mkdtemp()

    with open(f"{d}/gold.jsonl", "w") as f:
        for i, a_, b_ in zip(ids, gc, gr):
            f.write(json.dumps(dict(pair_id=i, gold_chosen=a_, gold_rejected=b_, group=i[:-1])) + "\n")

    for nm, sig in (("good", 0.5), ("noise", 50.0)):
        np.savez(f"{d}/{nm}.npz", pair_ids=ids[rng.permutation(n)] if False else ids,
                 chosen=gc[:, None] + sig * rng.normal(size=(n, K)),
                 rejected=gr[:, None] + sig * rng.normal(size=(n, K)))

    main(["--gold", f"{d}/gold.jsonl", "--proxy", f"good={d}/good.npz", f"noise={d}/noise.npz",
          "--diff", "good", "noise", "--B", "300"])


if __name__ == "__main__":
    import sys
    selftest() if len(sys.argv) > 1 and sys.argv[1] == "--selftest" else main()