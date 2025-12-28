#!/usr/bin/env python3
import re
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ---- Patterns that match YOUR logs ----
matrix_re = re.compile(r"^\s*MATRIX:\s*(\S+)")
tilek_re = re.compile(r"^\s*tile_k\s*=\s*(\d+)\s*$")

avg_header_re = re.compile(r"^=== Averages for (overlap|csr) on (\S+), cfg=\[(.*?)\] over \d+ runs ===")
avg_timing_re = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*([\d\.]+)\s*ms")

# We store per matrix + impl exactly one "best" record (you can change policy)
data = defaultdict(dict)  # data[matrix][impl] = record


def parse_file(path: str):
    with open(path, "r") as f:
        lines = f.readlines()

    matrix = None
    tile_k = None

    # first pass: find MATRIX and tile_k (tile_k appears per-run; take first)
    for line in lines:
        if matrix is None:
            m = matrix_re.search(line)
            if m:
                matrix = m.group(1)

        if tile_k is None:
            m = tilek_re.search(line)
            if m:
                tile_k = int(m.group(1))

        if matrix is not None and tile_k is not None:
            break

    # second pass: find the averages block and parse timings
    i = 0
    while i < len(lines):
        m = avg_header_re.match(lines[i].strip())
        if not m:
            i += 1
            continue

        impl = m.group(1)          # "overlap" or "csr"
        matrix2 = m.group(2)       # matrix name from header (should match)
        cfg = m.group(3).strip()

        # if MATRIX wasn’t found above, use this
        if matrix is None:
            matrix = matrix2

        record = {
            "matrix": matrix,
            "impl": impl,
            "cfg": cfg,
            "tile_k": tile_k,  # may be None if missing in file
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

            mt = avg_timing_re.match(line)
            if mt:
                key = mt.group(1)
                val = float(mt.group(2))

                if key == "End2End":
                    record["End2End"] = val
                elif key in record:
                    record[key] = val
                # Your file has: t_pure_computation = ...
                # We don't stack that directly; but if you want it too, add it here.

            i += 1

        # Keep best per matrix+impl:
        # If multiple files exist per matrix, keep the one with minimal End2End
        prev = data[matrix].get(impl)
        if prev is None or (record["End2End"] < prev["End2End"]):
            data[matrix][impl] = record

        # continue scanning in case file contains multiple avg blocks
        i += 1

    # Helpful debug print
    # print(f"Parsed {path} -> matrix={matrix}, tile_k={tile_k}, impls={list(data[matrix].keys())}")


def plot_all(outdir="plots_all"):
    os.makedirs(outdir, exist_ok=True)

    matrices = sorted(data.keys())
    if not matrices:
        print("No matrices parsed. Check your input path/glob.")
        return

    # which impls exist globally?
    all_impls = sorted({impl for m in matrices for impl in data[m].keys()})  # e.g. ["overlap"] or ["csr","overlap"]

    friendly_impl = {"csr": "Full GPU (CSR)", "overlap": "Overlap stream"}

    # ---- Plot 1: End2End for all matrices ----
    x = np.arange(len(matrices))
    n_impl = len(all_impls)
    total_width = 0.85
    width = total_width / max(n_impl, 1)

    plt.figure(figsize=(max(10, len(matrices) * 0.35), 5))
    for idx, impl in enumerate(all_impls):
        y = []
        for m in matrices:
            r = data[m].get(impl)
            y.append(np.nan if r is None else r["End2End"])
        offset = (idx - (n_impl - 1) / 2) * width
        plt.bar(x + offset, y, width=width, label=friendly_impl.get(impl, impl))

    # x tick label includes tile_k
    labels = []
    for m in matrices:
        tk = None
        # prefer overlap record tile_k, else any impl
        for impl in all_impls:
            rr = data[m].get(impl)
            if rr and rr.get("tile_k") is not None:
                tk = rr["tile_k"]
                break
        labels.append(f"{m}\nK={tk}" if tk is not None else m)

    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Average End-to-end time (ms)")
    plt.title("All matrices: End2End averages")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "all_matrices_end2end.png"), dpi=150)
    plt.close()

    # ---- Plot 2: Stacked breakdown for all matrices ----
    components = ["t_cpu_alloc", "t_gpu_alloc", "t_h2d_ms", "t_spmm_ms", "t_d2h_ms"]

    # choose consistent colors by component (matplotlib tab colors)
    component_colors = {
        "t_cpu_alloc": "tab:blue",
        "t_gpu_alloc": "tab:orange",
        "t_h2d_ms": "tab:red",
        "t_spmm_ms": "tab:purple",
        "t_d2h_ms": "tab:brown",
    }

    plt.figure(figsize=(max(10, len(matrices) * 0.35), 6))

    # We draw a stacked bar per matrix per impl (grouped), each stack is components.
    for impl_idx, impl in enumerate(all_impls):
        offset = (impl_idx - (n_impl - 1) / 2) * width
        bottom = np.zeros(len(matrices))

        # stack per component
        for comp in components:
            vals = []
            for m in matrices:
                r = data[m].get(impl)
                vals.append(0.0 if r is None else float(r.get(comp, 0.0)))
            vals = np.array(vals)

            # Only label component once (for legend cleanliness)
            label = comp if impl_idx == 0 else None

            plt.bar(
                x + offset,
                vals,
                width=width,
                bottom=bottom,
                label=label,
                color=component_colors[comp],
            )
            bottom += vals

    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Average time (ms)")
    plt.title("All matrices: Time breakdown (stacked)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Components")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "all_matrices_stacked.png"), dpi=150)
    plt.close()

    print(f"Wrote:\n  {outdir}/all_matrices_end2end.png\n  {outdir}/all_matrices_stacked.png")


def main():
    # CHANGE THIS GLOB to your folder
    files = sorted(glob.glob("results_best_tile_K/no_nsys/*"))
    if not files:
        print("No .txt files found under results_best_tile_k/no_nsys/")
        return

    for p in files:
        parse_file(p)

    plot_all()


if __name__ == "__main__":
    main()

