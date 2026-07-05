#!/usr/bin/env python3
"""Offline overlay + quantitative gate for the rung tracking-fidelity check.

Reads the CSV written by ocs2_target_publisher (mode=arc): observed base pose
(t,px,py,yaw) per tick, with a header carrying the commanded arc params. Fits a
circle to the observed path and compares its radius to the commanded arc_radius
-- the fidelity metric:

  R_fit ~= R_cmd     OCS2 followed the commanded SHAPE (gate PASS)
  R_fit >> R_cmd     OCS2 cut corners / ignored the curve (path too straight)
  R_fit  < R_cmd     OCS2 over-turned

This is rospy-free -- run on the host or in the container:
  python3 scripts/plot_tracking.py [csv_path]
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SETTLE_S = 2.0     # drop the initial standing/startup transient before fitting
TOL = 0.25         # |R_fit - R_cmd| / R_cmd gate


def fit_circle(x, y):
    """Algebraic (Kasa) least-squares circle fit -> (cx, cy, R)."""
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return cx, cy, float(np.sqrt(c + cx ** 2 + cy ** 2))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    csv = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(here, "..", "docs", "progress", "rung_track.csv")

    rows = []
    with open(csv) as f:
        header = f.readline()                       # "# mode=... arc_radius=..."
        f.readline()                                # "t,px,py,yaw" column row
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append([float(x) for x in line.split(",")])
    params = {k: float(v) for k, v in re.findall(r"(\w+)=([-+0-9.eE]+)", header)}
    R_cmd = params.get("arc_radius", float("nan"))
    arr = np.array(rows)
    t, px, py, yaw = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    keep = t >= (t[0] + SETTLE_S)
    px, py, t, yaw = px[keep], py[keep], t[keep], yaw[keep]
    arc_len = float(np.sum(np.hypot(np.diff(px), np.diff(py))))

    cx, cy, R_fit = fit_circle(px, py)
    err = abs(R_fit - R_cmd) / R_cmd if R_cmd else float("nan")
    subtend = arc_len / R_fit if R_fit > 1e-6 else 0.0    # radians of arc traced
    ok = (err < TOL) and (subtend > 0.35)                 # need enough arc to trust the fit

    fig, (axp, axy) = plt.subplots(1, 2, figsize=(13, 5))
    axp.plot(px, py, "k.-", lw=1.2, ms=3, label="observed path")
    axp.plot(px[0], py[0], "gd", ms=9, label="start (post-settle)")
    th = np.linspace(0, 2 * np.pi, 200)
    axp.plot(cx + R_fit * np.cos(th), cy + R_fit * np.sin(th), "r-", lw=1,
             label=f"fitted circle R={R_fit:.2f}")
    axp.plot(cx + R_cmd * np.cos(th), cy + R_cmd * np.sin(th), "b--", lw=1,
             label=f"commanded R={R_cmd:.2f}")
    axp.plot(cx, cy, "r+", ms=10)
    axp.set_aspect("equal"); axp.grid(True); axp.legend(fontsize=8, loc="best")
    axp.set_xlabel("x [m]"); axp.set_ylabel("y [m]")
    axp.set_title("Tracking fidelity  (%s, err=%.1f%%)"
                  % ("PASS" if ok else "FAIL", err * 100))

    axy.plot(t - t[0], np.degrees(np.unwrap(yaw)), "k-", lw=1, label="observed yaw")
    axy.grid(True); axy.legend(fontsize=8)
    axy.set_xlabel("t [s]"); axy.set_ylabel("yaw [deg]")
    axy.set_title("Base yaw over time")

    png = os.path.splitext(csv)[0] + ".png"
    fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)

    print("mode=%s  R_cmd=%.3f  R_fit=%.3f  err=%.1f%%  arc_len=%.2fm  "
          "subtend=%.2frad  n=%d"
          % (params.get("mode", "?") if "mode" in header else "?",
             R_cmd, R_fit, err * 100, arc_len, subtend, len(px)))
    print("wrote", png)
    print("GATE:", "PASS" if ok else "FAIL",
          "(need err<%.0f%% and subtend>0.35rad)" % (TOL * 100))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
