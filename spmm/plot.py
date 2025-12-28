#!/usr/bin/env python3
import re
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# --- Regex that matches your logs ---
matrix_re = re.compile(r"^\s*MATRIX:\s*(\S+)")
tilek_re = re.compile(r"^\s*tile_k\s*=\s*(\d+)\s*$")

avg_header_re = re.compile(
    r"^=== Averages for (overlap|csr) on (\S+), cfg=\[(.*?)\] over \d+ runs ==="
)

# Your per-run lines sometimes have "End2End=68009 ms" (no spaces)
timing_re_any = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=?\s*([\d\.]+)\s*ms\s*$")

data = defaultdict(dict)  # data[matrix][impl] = best_record


def parse_file(path: str):
    with open(path, "r") as f:
        lines = f.readlines()

    matrix = None
    tile_k = None

    # Find matrix name + first tile_k
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
            "tile_k": tile_k,  # may still be None
            "End2End": np.nan,
            "t_cpu_alloc": 0.0,
            "t_gpu_alloc": 0.0,
            "t_h2d_ms": 0.0,
            "t_spmm_ms": 0.0,
            "t_d2h_ms": 0.0,
            "t_pure_computation_and_transfers": np.nan,  # we’ll fill if present
        }

        i += 1
        while i < len(lines):
            l = lines[i].strip()
            if l == "" or l.startswith("==="):
                break

            mt = timing_re_any.match(l.replace("  ", " "))
            if mt:
                key = mt.group(1)
                val = float(mt.group(2))

                # Normalize a couple of key variants seen in logs
                if key == "t_pure_computation":
                    key = "t_pure_computation_and_transfers"
                # Some logs might say t_pure_computation_and_transfers already

                if key == "End2End":
                    record["End2End"] = val
                elif key in record:
                    record[key] = val

            i += 1

        # Keep best record per matrix+impl (min End2End)
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

    # Labels include tile_k
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

    # ---- Plot 0: tile_k per matrix ----
    plt.figure(figsize=(max(10, len(matrices) * 0.35), 4))
    plt.bar(x, tileks)
    plt.xticks(x, [m for m in matrices], rotation=90)
    plt.ylabel("tile_k (K)")
    plt.title("tile_k per matrix")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "all_matrices_tile_k.png"), dpi=150)
    plt.close()

    # ---- Plot 1: End2End ----
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

    # ---- Plot 2: Stacked breakdown (atomic components) ----
    components = ["t_cpu_alloc", "t_gpu_alloc", "t_h2d_ms", "t_spmm_ms", "t_d2h_ms"]
    component_colors = {
        "t_cpu_alloc": "tab:blue",
        "t_gpu_alloc": "tab:orange",
        "t_h2d_ms": "tab:red",
        "t_spmm_ms": "tab:purple",
        "t_d2h_ms": "tab:brown",
    }

    plt.figure(figsize=(max(10, len(matrices) * 0.35), 6))
    for impl_idx, impl in enumerate(all_impls):
        offset = (impl_idx - (n_impl - 1) / 2) * width
        bottom = np.zeros(len(matrices))

        for comp in components:
            vals = np.array([
                0.0 if data[m].get(impl) is None else float(data[m][impl].get(comp, 0.0))
                for m in matrices
            ])

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

    # ---- Plot 3: t_pure_computation_and_transfers ----
    # (separate plot because it overlaps/aggregates, not a clean stack component)
    plt.figure(figsize=(max(10, len(matrices) * 0.35), 5))
    for idx, impl in enumerate(all_impls):
        y = []
        for m in matrices:
            r = data[m].get(impl)
            if r is None:
                y.append(np.nan)
            else:
                y.append(r.get("t_pure_computation_and_transfers", np.nan))
        offset = (idx - (n_impl - 1) / 2) * width
        plt.bar(x + offset, y, width=width, label=friendly_impl.get(impl, impl))

    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Average time (ms)")
    plt.title("All matrices: t_pure_computation_and_transfers")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "all_matrices_pure_compute_and_transfers.png"), dpi=150)
    plt.close()

    print("Wrote plots to:", outdir)
    print(" - all_matrices_tile_k.png")
    print(" - all_matrices_end2end.png")
    print(" - all_matrices_stacked.png")
    print(" - all_matrices_pure_compute_and_transfers.png")


def main():
    # adjust to your actual folder name
    files = sorted(glob.glob("results_best_tile_K/no_nsys/*"))
    if not files:
        print("No .txt files found under results_best_tile_k/no_nsys/")
        return

    for p in files:
        parse_file(p)

    plot_all()


if __name__ == "__main__":
    main()

