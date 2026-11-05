"""scripts/check_kl_reach.py <g1_g3_summary.json>
Answers: where does 'max reachable KL 3.22' come from? Prints, per condition, the usable n grid
(truncated by kl_tol in choose_n_max), empirical vs analytic KL, and the analytic KL at full N."""

import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("summary")
a = ap.parse_args()
for name, s in json.load(open(a.summary)).items():
    n, x = np.asarray(s["n_grid"]), np.asarray(s["kl_emp_median"])
    ana, gap = np.asarray(s["kl_analytic"]), np.asarray(s["kl_gap_median"])
    N = s["g3"]["N"]

    print(f"[{name}] N={N}  n_max(usable)={s['n_max']}  max_kl_reached={s['max_kl_reached']:.2f}")
    print(f"   analytic BoN KL at N={N}: {np.log(N) - (N - 1) / N:.2f}   resampling cap ln N: {np.log(N):.2f}")
    print("   n      emp_KL  analytic  |gap|")

    for i in range(len(n)):
        print(f"   {n[i]:<6d} {x[i]:6.2f}  {ana[i]:7.2f}  {gap[i]:.3f}")

    print("   -> if n_max << N, the cap is the kl_tol truncation, not N itself\n")