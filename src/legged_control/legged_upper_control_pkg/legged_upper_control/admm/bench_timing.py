"""C5 timing harness — per-cycle ADMM planning time for the paper table.

Configs: python (golden), cpp sequential, cpp OpenMP-parallel; fleets N=3/4/5
(complete graph -> 3/6/10 edges). Straight-line reference, one obstacle, V-ish
lateral offsets. Reports mean/p50/p95 per-cycle wall time over M cycles after a
warm-up cycle, plus speedups. Writes bench_timing.csv next to this file.

Run (inside the SIL container, devel sourced):
    python3 legged_upper_control/admm/bench_timing.py [cycles]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants as C                      # noqa: E402
import admm_coordinator as py_ac           # noqa: E402
import admm_core_cpp                       # noqa: E402
from admm_core_cpp import coordinator as cpp_ac  # noqa: E402

CYCLES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OBSTACLES = [{"pos": (2.0, 0.2), "radius": 0.30}]


def fleet(n_dogs):
    dogs = tuple(range(1, n_dogs + 1))
    edges = tuple((a, b) for i, a in enumerate(dogs) for b in dogs[i + 1:])
    return dogs, edges


def scenario(dogs):
    xnow = {d: np.array([0.0, 1.0 * i, 0.0, 0.0]) for i, d in enumerate(dogs)}
    xdes = {}
    for i, d in enumerate(dogs):
        xd = np.zeros((C.N, 4))
        xd[:, 0] = np.linspace(0.2, 4.0, C.N)
        xd[:, 1] = 1.0 * i
        xdes[d] = xd
    return xnow, xdes


def bench(make_coord, dogs, edges):
    coord = make_coord(dogs, edges)
    xnow, xdes = scenario(dogs)
    coord.step(xnow, xdes)                      # warm-up (cold-start cycle)
    times = []
    for _ in range(CYCLES):
        t0 = time.perf_counter()
        coord.step(xnow, xdes)
        times.append((time.perf_counter() - t0) * 1e3)
    t = np.asarray(times)
    return dict(mean=float(t.mean()), p50=float(np.percentile(t, 50)),
                p95=float(np.percentile(t, 95)))


CONFIGS = {
    "python": lambda d, e: py_ac.ADMMCoordinator(dogs=d, edges=e,
                                                 obstacles=OBSTACLES),
    "cpp-seq": lambda d, e: cpp_ac.ADMMCoordinator(dogs=d, edges=e,
                                                   obstacles=OBSTACLES,
                                                   parallel=False),
    "cpp-par": lambda d, e: cpp_ac.ADMMCoordinator(dogs=d, edges=e,
                                                   obstacles=OBSTACLES,
                                                   parallel=True),
}


def main():
    rows = []
    for n_dogs in (3, 4, 5):
        dogs, edges = fleet(n_dogs)
        res = {}
        for name, mk in CONFIGS.items():
            res[name] = bench(mk, dogs, edges)
            rows.append((n_dogs, len(edges), name, res[name]))
        base, seq, par = res["python"], res["cpp-seq"], res["cpp-par"]
        print(f"N={n_dogs} ({len(edges)} edges, {CYCLES} cycles) "
              f"per-cycle mean/p50/p95 [ms]:")
        for name in CONFIGS:
            r = res[name]
            print(f"   {name:8s} {r['mean']:8.2f} / {r['p50']:8.2f} / {r['p95']:8.2f}")
        print(f"   speedup: cpp-seq vs python x{base['mean']/seq['mean']:.2f}, "
              f"cpp-par vs cpp-seq x{seq['mean']/par['mean']:.2f}, "
              f"cpp-par vs python x{base['mean']/par['mean']:.2f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bench_timing.csv")
    with open(out, "w") as f:
        f.write("n_dogs,n_edges,config,mean_ms,p50_ms,p95_ms\n")
        for n_dogs, ne, name, r in rows:
            f.write(f"{n_dogs},{ne},{name},{r['mean']:.4f},{r['p50']:.4f},"
                    f"{r['p95']:.4f}\n")
    print(f"[bench] wrote {out}")


if __name__ == "__main__":
    main()
