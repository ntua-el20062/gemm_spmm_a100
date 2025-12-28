#!/usr/bin/env python3
import re
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

matrix_re = re.compile(r"^\s*MATRIX:\s*(\S+)")
tilek_re = re.compile(r"^\s*tile_k\s*=\s*(\d+)\s*$")

avg_header_re = re.compile(
    r"^=== Averages for (overlap|csr) on (\S+), cfg=\[(.*?)\] over \d+ runs ==="
)


timing_re_any = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=?\s*([\d\.]+)\s*ms\s*$")


data = defaultdict(dict)


def parse_file(path: str):
    with open(path, "r") as f:
        lines = f.readlines()

    matrix = None
    tile_k = None

 
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

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        mh = avg_header_re.match(line)
        if not mh:
            i += 1
            continue

        impl = mh.group(1)      # overlap / csr
        matrix2 = mh.group(2)   # matrix name in header
        cfg = mh.group(3).strip()

        if matrix is None:
            matrix = matrix2

        record = {
            "matrix": matrix,
            "impl": impl,
            "cfg": cfg,
            "tile_k": tile_k,  # may be None if missing
            "End2End": np.nan,
            "t_cpu_alloc": 0.0,
            "t_gpu_alloc": 0.0,
            "t_h2d_ms": 0.0,
            "t_spmm_ms": 0.0,
            "t_d2h_ms": 0.0,
            "t_pure_computation_and_transfers": 0.0,
        }

        i += 1
        while i < len(lines):
            l = lines[i].strip()
            if l == "" or l.startswith("==="):
                break

            mt = timing_re_any.match(l)
            if mt:
                key = mt.group(1)
                val = float(mt.group(2))

  
                if key == "t_pure_computation":
                    key = "t_pure_computation_and_transfers"

                if key == "End2End":
                    record["End2End"] = val
                elif key in record:
                    record[key] = val

            i += 1

   
        prev = data[matrix].get(impl)
        if prev is None or (record["End2End"] < prev["End2End"]):
            data[matrix][impl] = record

        i += 1


def plot_all(outdir="plots_all"):
    os.makedirs(outdir, exist_ok=True)

    matrices = sorted(data.keys())
    if not matrices:
        print("No matrices parsed. Check your input glob.")
        return

    all_impls = sorted({impl for m in matrices for impl in data[m].keys()})
    friendly_impl = {"csr": "Full GPU (CSR)", "overlap": "Overlap stream"}

    x = np.arange(len(matrices))
    n_impl = len(all_impls)
    total_width = 0.85
    width = total_width / max(n_impl, 1)

    
    labels = []
    tileks = []
    for m in matrices:
        tk = None
        for impl in all_impls:
            r = data[m].get(impl)
            if r and r.get("tile_k") is not None:
                tk = r["tile_k"]
                break
        tileks.append(np.nan if tk is None else tk)
        labels.append(f"{m}\nK={tk}" if tk is not None else m)

    plt.figure(figsize=(max(10, len(matrices) * 0.35), 4))
    plt.bar(x, tileks)
    plt.xticks(x, [m for m in matrices], rotation=90)
    plt.ylabel("tile_k (K)")
    plt.title("tile_k per matrix")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "all_matrices_tile_k.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(max(10, len(matrices) * 0.35), 5))
    for idx, impl in enumerate(all_impls):
        y = [np.nan if data[m].get(impl) is None else data[m][impl]["End2End"] for m in matrices]
        offset = (idx - (n_impl - 1) / 2) * width
        plt.bar(x + offset, y, width=width, label=friendly_impl.get(impl, impl))

    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Average End-to-end time (ms)")
    plt.title("All matrices: End2End averages")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "all_matrices_end2end.png"), dpi=150)
    plt.close()

    components = [
        "t_cpu_alloc",
        "t_gpu_alloc",
        "t_pure_computation_and_transfers",
        "t_other",
    ]
    component_colors = {
        "t_cpu_alloc": "tab:blue",
        "t_gpu_alloc": "tab:orange",
        "t_pure_computation_and_transfers": "tab:green",
        "t_other": "tab:gray",
    }

    plt.figure(figsize=(max(10, len(matrices) * 0.35), 6))

    for impl_idx, impl in enumerate(all_impls):
        offset = (impl_idx - (n_impl - 1) / 2) * width
        bottom = np.zeros(len(matrices))

        end2end = np.array([
            np.nan if data[m].get(impl) is None else float(data[m][impl].get("End2End", np.nan))
            for m in matrices
        ])
        cpu = np.array([
            0.0 if data[m].get(impl) is None else float(data[m][impl].get("t_cpu_alloc", 0.0))
            for m in matrices
        ])
        gpu = np.array([
            0.0 if data[m].get(impl) is None else float(data[m][impl].get("t_gpu_alloc", 0.0))
            for m in matrices
        ])
        pure = np.array([
            0.0 if data[m].get(impl) is None else float(
                data[m][impl].get("t_pure_computation_and_transfers", 0.0)
            )
            for m in matrices
        ])

        other = end2end - (cpu + gpu + pure)
        other = np.where(np.isnan(other), 0.0, other)
        other = np.maximum(other, 0.0)

        stacks = {
            "t_cpu_alloc": cpu,
            "t_gpu_alloc": gpu,
            "t_pure_computation_and_transfers": pure,
            "t_other": other,
        }

        for comp in components:
            vals = stacks[comp]
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
    plt.title("All matrices: End2End breakdown (cpu + gpu + pure + other)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Components")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "all_matrices_stacked.png"), dpi=150)
    plt.close()

    print("Wrote plots to:", outdir)
    print(" - all_matrices_tile_k.png")
    print(" - all_matrices_end2end.png")
    print(" - all_matrices_stacked.png")


def main():
    files = sorted(glob.glob("results_best_tile_K/no_nsys/*.txt"))
    if not files:
        print("No .txt files found under results_best_tile_k/no_nsys/")
        return

    for p in files:
        parse_file(p)

    plot_all()


if __name__ == "__main__":
    main()

