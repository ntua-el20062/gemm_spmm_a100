#!/usr/bin/env python3
import re
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Regex ----------------
matrix_re = re.compile(r"^\s*MATRIX:\s*(\S+)")
tilek_re = re.compile(r"^\s*tile_k\s*=\s*(\d+)\s*$")

avg_header_re = re.compile(
    r"^=== Averages for (overlap|csr) on (\S+), cfg=\[(.*?)\] over \d+ runs ==="
)

timing_re = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=?\s*([\d\.]+)\s*ms\s*$")

# data[matrix][impl] = record
data = defaultdict(dict)


def parse_file(path):
    with open(path, "r") as f:
        lines = f.readlines()

    matrix = None
    tile_k = None

    # Get matrix name + tile_k
    for line in lines:
        if matrix is None:
            m = matrix_re.search(line)
            if m:
                matrix = m.group(1)
        if tile_k is None:
            m = tilek_re.search(line)
            if m:
                tile_k = int(m.group(1))
        if matrix and tile_k:
            break

    i = 0
    while i < len(lines):
        mh = avg_header_re.match(lines[i].strip())
        if not mh:
            i += 1
            continue

        impl = mh.group(1)   # overlap / csr
        matrix2 = mh.group(2)

        if matrix is None:
            matrix = matrix2

        record = {
            "matrix": matrix,
            "impl": impl,
            "tile_k": tile_k,
            "End2End": np.nan,
            "t_cpu_alloc": 0.0,
            "t_gpu_alloc": 0.0,
            "t_h2d_ms": 0.0,
            "t_spmm_ms": 0.0,
            "t_d2h_ms": 0.0,
        }

        i += 1
        while i < len(lines):
            line = lines[i].strip()
            if line == "" or line.startswith("==="):
                break

            mt = timing_re.match(line)
            if mt:
                key, val = mt.group(1), float(mt.group(2))
                if key == "End2End":
                    record["End2End"] = val
                elif key in record:
                    record[key] = val
            i += 1

        # keep best (min End2End)
        prev = data[matrix].get(impl)
        if prev is None or record["End2End"] < prev["End2End"]:
            data[matrix][impl] = record

        i += 1


def plot_all(outdir="plots_all"):
    os.makedirs(outdir, exist_ok=True)

    matrices = sorted(data.keys())
    impls = sorted({impl for m in matrices for impl in data[m]})

    x = np.arange(len(matrices))
    width = 0.8 / max(len(impls), 1)

    labels = []
    tileks = []
    for m in matrices:
        r = next(iter(data[m].values()))
        labels.append(f"{m}\nK={r['tile_k']}")
        tileks.append(r["tile_k"])

    # ---------- tile_k plot ----------
    plt.figure(figsize=(max(10, len(matrices) * 0.35), 4))
    plt.bar(x, tileks)
    plt.xticks(x, matrices, rotation=90)
    plt.ylabel("tile_k")
    plt.title("tile_k per matrix")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/tile_k.png", dpi=150)
    plt.close()

    # ---------- End2End ----------
    plt.figure(figsize=(max(10, len(matrices) * 0.35), 5))
    for i, impl in enumerate(impls):
        y = [data[m][impl]["End2End"] if impl in data[m] else np.nan for m in matrices]
        plt.bar(x + (i - 0.5) * width, y, width, label=impl)

    plt.xticks(x, labels, rotation=90)
    plt.ylabel("End2End (ms)")
    plt.title("End-to-end time")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/end2end.png", dpi=150)
    plt.close()

    # ---------- STACKED TIMERS (NO PURE COMPUTE) ----------
    components = [
        "t_cpu_alloc",
        "t_gpu_alloc",
        "t_h2d_ms",
        "t_spmm_ms",
        "t_d2h_ms",
    ]

    colors = {
        "t_cpu_alloc": "tab:blue",
        "t_gpu_alloc": "tab:orange",
        "t_h2d_ms": "tab:red",
        "t_spmm_ms": "tab:purple",
        "t_d2h_ms": "tab:brown",
    }

    plt.figure(figsize=(max(10, len(matrices) * 0.35), 6))

    for i, impl in enumerate(impls):
        bottom = np.zeros(len(matrices))
        offset = (i - 0.5) * width

        for comp in components:
            vals = [
                data[m][impl][comp] if impl in data[m] else 0.0
                for m in matrices
            ]
            plt.bar(
                x + offset,
                vals,
                width,
                bottom=bottom,
                color=colors[comp],
                label=comp if i == 0 else None,
            )
            bottom += np.array(vals)

    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Time (ms)")
    plt.title("Stacked timers (cpu, gpu, h2d, spmm, d2h)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Timers")
    plt.tight_layout()
    plt.savefig(f"{outdir}/stacked_timers.png", dpi=150)
    plt.close()

    print("Plots written to:", outdir)


def main():
    files = glob.glob("results_best_tile_K/no_nsys/*.txt")
    if not files:
        print("No input files found.")
        return

    for f in files:
        parse_file(f)

    plot_all()


if __name__ == "__main__":
    main()

