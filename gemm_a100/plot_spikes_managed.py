#!/usr/bin/env python3
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker   # <-- ADD

log_path = "results_%_managed.txt"
out_dir = "plots"
spikes = [0, 250, 500, 750, 1000]


def split_runs(text: str):
    parts = re.split(r"\bMANAGED VERSION\b", text)
    return [p.strip() for p in parts if p.strip()]


def parse_run(run_text: str):
    m = re.search(r"Running GEMM with m=(\d+),\s*n=(\d+),\s*k=(\d+)", run_text)
    if not m:
        return None
    mnk = (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    times = {}
    for it_s, ms_s in re.findall(r"iter:\s*(\d+)\s*,\s*t_iter:\s*([0-9]*\.?[0-9]+)\s*ms", run_text):
        times[int(it_s)] = float(ms_s)

    if not times:
        return None

    xs = np.array(sorted(times.keys()), dtype=int)
    ys = np.array([times[i] for i in xs], dtype=float)
    return {"mnk": mnk, "x": xs, "y": ys}


def values_at_spikes(x, y, spikes_list):
    lookup = dict(zip(x, y))
    sx, sy = [], []
    for s in spikes_list:
        if s in lookup:
            sx.append(s)
            sy.append(lookup[s])
    return sx, sy


with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

runs = []
for rt in split_runs(text):
    r = parse_run(rt)
    if r is not None:
        runs.append(r)

if not runs:
    raise RuntimeError("No MALLOC runs parsed. Check the log format / filename.")

by_size = {}
for r in runs:
    by_size.setdefault(r["mnk"], []).append(r)

os.makedirs(out_dir, exist_ok=True)

for (m, n, k), runs_same_size in sorted(by_size.items()):
    fig, ax = plt.subplots(figsize=(12, 5))  # <-- CHANGE: use ax

    for idx, r in enumerate(runs_same_size, start=1):
        x, y = r["x"], r["y"]
        ax.plot(x, y, linewidth=2.0, alpha=0.85, label=f"Run {idx}")

        sx, sy = values_at_spikes(x, y, spikes)
        ax.scatter(sx, sy, s=70, zorder=4)

    for s in spikes:
        ax.axvline(s, linestyle="--", linewidth=1, color="gray", zorder=0)

    max_iter = max(int(r["x"].max()) for r in runs_same_size)
    xticks = sorted(set([0] + spikes + [max_iter]))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])

    ax.set_xlabel("Iteration (only spikes labeled)")
    ax.set_ylabel("t_iter (ms)")
    ax.set_title(f"GEMM iteration time (Managed) | m={m}, n={n}, k={k}")
    ax.grid(alpha=0.3)
    ax.legend()

    all_y = np.concatenate([r["y"] for r in runs_same_size])
    if np.all(all_y > 0):
        # log2 scale
        ax.set_yscale("log", base=2)

        # powers-of-2 major ticks
        ax.yaxis.set_major_locator(mticker.LogLocator(base=2))

        # labels as plain numbers: 512, 1024, 2048...
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v)}" if v >= 1 else f"{v:g}")
        )

        # hide minor labels (optional)
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    out_png = os.path.join(out_dir, f"iters_managed_m{m}_n{n}_k{k}.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"[OK] Saved: {out_png}")

