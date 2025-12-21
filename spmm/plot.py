#!/usr/bin/env python3
import re
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

data = defaultdict(lambda: defaultdict(lambda: {"records": []}))

overlap_header_re = re.compile(
    r"=== Averages for overlap on (\S+), cfg=\[(.*?)\] over \d+ runs ==="
)
csr_header_re = re.compile(
    r"=== Averages for csr on (\S+), cfg=\[(.*?)\] over \d+ runs ==="
)

timing_re = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*([\d\.]+)\s*ms")


def parse_file(path: str):
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        m_ov = overlap_header_re.search(line)
        m_csr = csr_header_re.search(line)

        if m_ov:
            matrix = m_ov.group(1)
            cfg_str = m_ov.group(2).strip()
            impl = "overlap"
        elif m_csr:
            matrix = m_csr.group(1)
            cfg_str = m_csr.group(2).strip()
            impl = "csr"
        else:
            i += 1
            continue

        try:
            K = int(cfg_str.split()[0])
        except ValueError:
            mK = re.search(r"\d+", cfg_str)
            K = int(mK.group(0)) if mK else -1

        record = {
            "K": K,
            "cfg": cfg_str,
            "End2End": np.nan,
            "t_gpu_alloc": 0.0,
            "t_cpu_alloc": 0.0,
            "t_h2d_ms": 0.0,
            "t_spmm_ms": 0.0,
            "t_d2h_ms": 0.0,
            "t_pure_computation_and_transfers": 0.0,

        }

        i += 1
        while i < len(lines):
            l2 = lines[i].strip()
            if (
                l2 == ""
                or l2.startswith("---")
                or l2.startswith("Running ")
                or l2.startswith("Skipping average computation")
            ):
                break

            m_t = timing_re.match(lines[i])
            if m_t:
                key = m_t.group(1)
                val = float(m_t.group(2))
                if key == "End2End":
                    record["End2End"] = val
                elif key in record:
                    record[key] = val

            i += 1

        data[matrix][impl]["records"].append(record)

    print(f"Parsed {path}")


def pick_best_per_K(records):
    """
    Group records by K and keep the one with minimal End2End for each K.
    Returns dict: K -> record
    """
    best = {}
    for r in records:
        K = r["K"]
        if K not in best or (r["End2End"] < best[K]["End2End"]):
            best[K] = r
    return best


def stacked_arrays(best_by_K, Ks, components):
    """
    For a dict best_by_K and a list of Ks, return values per component.
    Returns dict: comp -> list over Ks
    """
    out = {c: [] for c in components}
    for K in Ks:
        r = best_by_K.get(K)
        for c in components:
            out[c].append(0.0 if r is None else r.get(c, 0.0))
    return out


friendly_impl_name = {
    "csr": "Full GPU (CSR)",
    "overlap": "Overlap stream",
}


def plot_matrix(matrix: str, impl_data, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)

    implementations = sorted(impl_data.keys())  # e.g. ["csr", "overlap"]
    if not implementations:
        print(f"No implementations for {matrix}, skipping.")
        return

    best = {}
    all_Ks = set()
    for impl in implementations:
        recs = impl_data[impl]["records"]
        if not recs:
            continue
        best_impl = pick_best_per_K(recs)
        best[impl] = best_impl
        all_Ks.update(best_impl.keys())

    if not all_Ks:
        print(f"No records for {matrix}, skipping.")
        return

    Ks = sorted(all_Ks)
    x = np.arange(len(Ks))

    n_impl = len(implementations)
    total_width = 0.8
    width = total_width / max(n_impl, 1)

    plt.figure(figsize=(8, 5))
    for idx, impl in enumerate(implementations):
        best_impl = best[impl]
        y = [best_impl.get(K, {}).get("End2End", np.nan) for K in Ks]
        offset = (idx - (n_impl - 1) / 2) * width
        plt.bar(
            x + offset,
            y,
            width=width,
            label=friendly_impl_name.get(impl, impl),
        )

    plt.xticks(x, [str(K) for K in Ks])
    plt.xlabel("K")
    plt.ylabel("Average End-to-end time (ms)")
    plt.title(f"{matrix}: End-to-end (Average)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{matrix}_end2end.png"), dpi=150)
    plt.close()


    plt.figure(figsize=(8, 5))
    for idx, impl in enumerate(implementations):
        best_impl = best[impl]
        y = [best_impl.get(K, {}).get("t_pure_computation_and_transfers", np.nan) for K in Ks]
        offset = (idx - (n_impl - 1) / 2) * width
        plt.bar(
            x + offset,
            y,
            width=width,
            label=friendly_impl_name.get(impl, impl),
        )

    plt.xticks(x, [str(K) for K in Ks])
    plt.xlabel("K")
    plt.ylabel("Average Compute time, (overlaped computation and spmm, ms)")
    plt.title(f"{matrix}: Compute time (Average)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{matrix}_compute.png"), dpi=150)
    plt.close()

    components = ["t_gpu_alloc", "t_cpu_alloc", "t_h2d_ms", "t_spmm_ms", "t_d2h_ms"]

    component_colors = {
        "t_gpu_alloc": "tab:orange",
        "t_cpu_alloc": "tab:green",
        "t_h2d_ms": "tab:red",
        "t_spmm_ms": "tab:purple",
        "t_d2h_ms": "tab:brown",
    }

    plt.figure(figsize=(10, 6))
    for impl_idx, impl in enumerate(implementations):
        best_impl = best[impl]
        stack = stacked_arrays(best_impl, Ks, components)
        offset = (impl_idx - (n_impl - 1) / 2) * width
        bottom = np.zeros(len(Ks))

        for comp in components:
            vals = np.array(stack[comp])
            if np.all(vals == 0.0):
                continue

            label = comp if impl_idx == 0 else None  # label once per component
            plt.bar(
                x + offset,
                vals,
                width=width,
                bottom=bottom,
                label=label,
                color=component_colors[comp],
            )
            bottom += vals

    plt.xticks(x, [str(K) for K in Ks])
    plt.xlabel("K")
    plt.ylabel("Average Time (ms)")
    plt.title(f"{matrix}: Time Breakdown (Stacked)")
    plt.grid(axis="y", alpha=0.3)

    plt.legend(title="Components")

    ax = plt.gca()
    if "csr" in implementations and "overlap" in implementations:
        note = "Implementation:\nleft: Full GPU (CSR)\nright: Overlap stream"
    elif "csr" in implementations:
        note = "Implementation:\nFull GPU (CSR)"
    elif "overlap" in implementations:
        note = "Implementation:\nOverlap stream"
    else:
        note = ""

    if note:
        ax.text(
            0.98,
            0.98,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", alpha=0.1),
        )

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{matrix}_stacked.png"), dpi=150)
    plt.close()

    print(f"Plots written for {matrix}")


def main():
    files = sorted(glob.glob("results_best_tile_K/*.txt"))
    for path in files:
        parse_file(path)

    for matrix, impls in data.items():
        plot_matrix(matrix, impls)


if __name__ == "__main__":
    main()

