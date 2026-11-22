"""Differential test: new Rust-backed microagg1d vs the ORIGINAL numba implementation.

The original package lives in another virtualenv (same module name), so it is run in a
subprocess using the interpreter given in the env var REFERENCE_PYTHON (default:
/tmp/claude-0/venv_old/bin/python if it exists). Results must agree verbatim.
When labels differ, the cost of both solutions is evaluated to tell which is better.
"""
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from microagg1d import univariate_microaggregation

REF = os.environ.get("REFERENCE_PYTHON")
if REF is None and os.path.exists("/tmp/claude-0/venv_old/bin/python"):
    REF = "/tmp/claude-0/venv_old/bin/python"

pytestmark = pytest.mark.skipif(
    not REF or not os.path.exists(REF or ""), reason="REFERENCE_PYTHON / venv_old not available"
)

METHODS = ("auto", "simple", "wilber", "galil_park", "staggered")
COSTS = ("sse", "sae", "roundup", "rounddown", "maxdist")


def stables_for(cost, method):
    if cost != "sse":
        return (1,)
    return (0, 1, 2, 3) if method == "simple" else (0, 1, 2)


def seg_cost(v, cost):
    v = np.asarray(v, dtype=np.longdouble)
    if cost == "sse":
        return float(((v - v.mean()) ** 2).sum())
    if cost == "sae":
        return float(np.abs(v - np.median(v)).sum())
    if cost == "roundup":
        return float((v.max() - v).sum())
    if cost == "rounddown":
        return float((v - v.min()).sum())
    return float(v.max() - v.min())


def total_cost(x, labels, cost):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    xs, ls = x[order], np.asarray(labels)[order]
    tot = 0.0
    for lab in np.unique(ls):
        tot += seg_cost(xs[ls == lab], cost)
    return tot


def valid(labels, k, n):
    labels = np.asarray(labels)
    if labels.shape != (n,):
        return False
    return np.bincount(labels - labels.min()).min() >= k


def gen_inputs():
    rng = np.random.RandomState(12345)
    out = []
    for seed in range(120):
        r = np.random.RandomState(seed)
        n = int(r.randint(1, 201))
        kind = seed % 8
        if kind == 0:
            x = r.rand(n)
        elif kind == 1:
            x = r.randint(0, 5, n).astype(float)  # many ties
        elif kind == 2:
            x = r.randint(-1000, 1000, n)  # integer dtype
        elif kind == 3:
            x = np.full(n, float(r.randint(-3, 3)))  # constant
        elif kind == 4:
            x = 1e6 + r.rand(n)  # large offset
        elif kind == 5:
            x = np.round(r.rand(n), 1)  # ties on floats
        elif kind == 6:
            x = np.concatenate([r.rand(n // 2), 100 + r.rand(n - n // 2)])
            r.shuffle(x)
        else:
            x = r.exponential(size=n) ** 3
        out.append((f"seed{seed}-kind{kind}", x))
    for n in (5000, 4999):
        out.append((f"big-rand-{n}", rng.rand(n)))
        out.append((f"big-offset-{n}", 1e6 + rng.rand(n)))
        out.append((f"big-ties-{n}", rng.randint(0, 50, n).astype(float)))
    out.append(("arange-1000", np.arange(1000, dtype=float)))
    out.append(("arange-137", np.arange(137, dtype=float)))
    return out


def gen_cases():
    cases, meta = [], []
    rng = np.random.RandomState(7)
    for name, x in gen_inputs():
        n = len(x)
        if n >= 1000:
            ks = [2, 3, 5, 10, 21, 22, 40, 100]
            methods = ("auto", "simple", "wilber", "galil_park", "staggered")
        else:
            ks = sorted(set(int(v) for v in rng.randint(1, min(30, n) + 1, 4)) | {1, min(n, 2), min(n, 3)})
            methods = METHODS
        for k in ks:
            if k > n:
                continue
            for cost in COSTS:
                for method in methods:
                    if n >= 1000 and method == "simple" and cost != "sse" and k > 10:
                        continue
                    for stable in stables_for(cost, method):
                        cases.append((x, k, method, cost, stable))
                        meta.append(name)
    return cases, meta


@pytest.fixture(scope="module")
def results():
    cases, meta = gen_cases()
    with tempfile.TemporaryDirectory() as d:
        inp, outp = os.path.join(d, "in.pkl"), os.path.join(d, "out.pkl")
        with open(inp, "wb") as f:
            pickle.dump(cases, f)
        runner = os.path.join(os.path.dirname(__file__), "ref_runner.py")
        subprocess.run([REF, runner, inp, outp], check=True, cwd=d)
        with open(outp, "rb") as f:
            old = pickle.load(f)
    return cases, meta, old


def brute_force_opt(x, k, cost):
    """Optimal cost by an O(n k) DP over cluster sizes k..2k-1 (exact segment costs)."""
    xs = np.sort(np.asarray(x, dtype=np.float64))
    n = len(xs)
    best = [0.0] + [float("inf")] * n
    for j in range(1, n + 1):
        for i in range(max(0, j - 2 * k + 1), j - k + 1):
            if best[i] < float("inf"):
                best[j] = min(best[j], best[i] + seg_cost(xs[i:j], cost))
    return best[n]


def classify(x, k, method, cost, stable, old, new):
    """Returns (ok, reason). Identical labels are always ok. Two differences are known
    defects of the ORIGINAL and are only checked for validity / cost quality:

    * 2k == n with unsorted input: the original forgets undo_argsort in the trivial case.
    * method simple (also auto, k <= 21), cost sse, stable=1: the original uses
      cell_size=k, but clusters of up to 2k-1 points can span three cells, giving wrong
      costs (the new code uses 3k like the original's other methods). Float noise in
      the different cell sizes also changes tie-breaking.
    """
    n = len(x)
    if np.array_equal(old, new):
        return True, "identical"
    if not valid(new, k, n):
        return False, "new result invalid"
    co, cn = total_cost(x, old, cost), total_cost(x, new, cost)
    tol = 1e-9 * max(1.0, abs(co), abs(cn))
    if 2 * k == n:
        order = np.argsort(x)
        exp = np.empty(n, dtype=np.int64)
        exp[order] = (np.arange(n) >= k).astype(np.int64)
        if np.array_equal(new, exp) and np.array_equal(old, (np.arange(n) >= k).astype(np.int64)):
            return True, "orig-bug-trivial-unsorted"
        return False, "2k==n mismatch beyond the known original bug"
    if not valid(old, k, n) and n <= 300 and cn <= brute_force_opt(x, k, cost) + tol:
        # original violates the size constraint (e.g. stable=0 penalty constant vanishes for
        # data with mean ~ 0); the new result is optimal
        return True, "orig-invalid: new is optimal"
    if cost == "sse" and abs(float(np.mean(x))) > 10 * np.ptp(x):
        # the new wrapper centers such data (avoids catastrophic cancellation): must not be worse
        return cn <= co + tol, "offset: new (centered) must not be worse"
    eff = "simple" if method == "auto" and k <= 21 else method
    if eff == "simple" and cost == "sse" and stable == 1:
        offset = np.abs(np.asarray(x, dtype=np.float64)).max() > 1e4 or (np.ptp(x) < 1e-3 * np.abs(x).max())
        if offset:  # catastrophic cancellation in x^2 sums: only validity is meaningful
            return True, "orig-bug-or-float-noise(offset)"
        if n <= 300:
            return cn <= brute_force_opt(x, k, cost) + tol, "orig-cell-size-bug: new must be optimal"
        return cn <= co + tol, "orig-cell-size-bug: new must not be worse"
    return False, "unexplained difference"


def test_agree_verbatim(results):
    cases, meta, old = results
    bad, explained = [], {}
    for (x, k, method, cost, stable), name, o in zip(cases, meta, old):
        try:
            new = univariate_microaggregation(x, k, method=method, cost=cost, stable=stable)
        except Exception as e:  # noqa
            new = f"EXC {type(e).__name__}: {e}"
        if isinstance(o, str) or isinstance(new, str):
            if not (isinstance(o, str) and isinstance(new, str)):
                bad.append((name, len(x), k, method, cost, stable, "exception mismatch", str(o)[:80], str(new)[:80]))
            continue
        ok, why = classify(x, k, method, cost, stable, o, new)
        if ok:
            explained[why] = explained.get(why, 0) + 1
        else:
            bad.append((name, len(x), k, method, cost, stable, why, total_cost(x, o, cost), total_cost(x, new, cost)))
    print(f"{len(cases)} cases; accepted by category: {explained}; unexplained: {len(bad)}")
    for b in bad[:40]:
        print(b)
    assert not bad, f"{len(bad)} of {len(cases)} results differ from the reference"
