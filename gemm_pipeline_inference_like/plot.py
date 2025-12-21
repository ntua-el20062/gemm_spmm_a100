import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

RESULTS_FILE = "results_all_impl_compared.txt"

# Names of implementations you use in the logs
impl_names = ["INITIAL APPROACH", "DOUBLE BUFFERING", "FULL OVERLAP"]

# results[config_id][impl] = { metric_name: value }
results = defaultdict(lambda: defaultdict(dict))
# config_meta[config_id] = dict(N=..., B=..., steps=...)
config_meta = {}


# ----------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------

def parse_metric_line(line):
    line = line.strip()

    # GEMM (either old full text or new t_gemm)
    m = re.search(r"(Total gemm\+norm.*=|total gemm\+norm time.*=|t_gemm\s*=\s*)([0-9.eE+-]+)", line)
    if m:
        return "t_gemm", float(m.group(2))

    # H2D
    m = re.search(r"(total h2d memcpy time.*=|total H2D memcpy time.*=|t_h2d\s*=\s*)([0-9.eE+-]+)", line)
    if m:
        return "t_h2d", float(m.group(2))

    # GPU alloc
    m = re.search(r"(t_gpu_alloc|t_alloc_gpu)\s*=\s*([0-9.eE+-]+)", line)
    if m:
        return "t_alloc_gpu", float(m.group(2))

    # CPU alloc
    m = re.search(r"(t_cpu_alloc|t_alloc_cpu)\s*=\s*([0-9.eE+-]+)", line)
    if m:
        return "t_alloc_cpu", float(m.group(2))

    # init
    m = re.search(r"(t_init)\s*=\s*([0-9.eE+-]+)", line)
    if m:
        return "t_init", float(m.group(2))

    # end-to-end
    m = re.search(r"(t_end_2_end)\s*=\s*([0-9.eE+-]+)", line)
    if m:
        return "t_end_2_end", float(m.group(2))

    # D2H (parsed but not used in stacked plots)
    m = re.search(r"(t_d2h)\s*=\s*([0-9.eE+-]+)", line)
    if m:
        return "t_d2h", float(m.group(2))

    return None, None


def parse_config_line(line):
    """
    Parse lines like:
      'Using batch_size = 5 (num_steps = 40)'
      'Using N = 25000, buff_size = 5 (num_steps = 40)'
      'Using N = 25000, batch_size = 5 (num_steps = 40)'
    Return (config_id, meta_dict) or (None, None).
    """
    line = line.strip()

    # Pattern with N and (batch|buff)
    m = re.search(
        r"Using\s+N\s*=\s*(\d+)\s*,\s*(?:batch_size|buff_size)\s*=\s*(\d+)\s*\(num_steps\s*=\s*(\d+)\)",
        line
    )
    if m:
        N = int(m.group(1))
        B = int(m.group(2))
        steps = int(m.group(3))
        config_id = f"N={N}_B={B}_steps={steps}"
        return config_id, {"N": N, "B": B, "steps": steps}

    # Pattern without N (older prints: only batch_size + num_steps)
    m = re.search(
        r"Using\s+(?:batch_size|buff_size)\s*=\s*(\d+)\s*\(num_steps\s*=\s*(\d+)\)",
        line
    )
    if m:
        B = int(m.group(1))
        steps = int(m.group(2))
        N = None
        config_id = f"N=NA_B={B}_steps={steps}"
        return config_id, {"N": None, "B": B, "steps": steps}

    return None, None


def parse_results(filename):
    current_impl = None
    current_config_id = None

    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip()

            # Detect which implementation block we're in
            for impl in impl_names:
                if impl in line:
                    current_impl = impl

            # Detect configuration line
            if "Using " in line and "num_steps" in line:
                cfg_id, meta = parse_config_line(line)
                if cfg_id is not None:
                    current_config_id = cfg_id
                    config_meta[cfg_id] = meta

            # detect metric line
            metric, val = parse_metric_line(line)
            if metric and current_impl and current_config_id:
                results[current_config_id][current_impl][metric] = val


# ----------------------------------------------------------
# Plot functions
# ----------------------------------------------------------

def plot_e2e_per_config():
    """
    For each (N, B, steps) configuration, plot end-to-end times
    for all implementations (including those that didn't run, shown as 0).
    """
    for cfg_id in sorted(results.keys()):
        impls = impl_names

        # skip only if *no* impl has metrics for this config
        if all(impl not in results[cfg_id] for impl in impls):
            continue

        y = []
        for impl in impls:
            metrics = results[cfg_id].get(impl, {})
            y.append(metrics.get("t_end_2_end", 0.0))

        meta = config_meta.get(cfg_id, {})
        N = meta.get("N", "NA")
        B = meta.get("B", "NA")
        steps = meta.get("steps", "NA")

        plt.figure()
        x = np.arange(len(impls))
        plt.bar(x, y)
        plt.xticks(x, impls, rotation=20)
        plt.ylabel("Time (ms)")
        plt.title(f"End-to-End Time (N={N}, B={B}, num_steps={steps})")
        plt.tight_layout()
        out_name = f"e2e_{cfg_id}.png".replace("=", "").replace(",", "_")
        plt.savefig(out_name, dpi=200)
        plt.close()


def plot_stacked_timers_per_config():
    """
    For each configuration, create a stacked bar plot of:
      t_alloc_gpu, t_alloc_cpu, t_init, t_h2d, t_gemm
    for each implementation. Implementations that didn't run are shown
    with all-zero segments. t_d2h is excluded.
    """
    timer_order = ["t_alloc_gpu", "t_alloc_cpu", "t_init", "t_h2d", "t_gemm"]

    for cfg_id in sorted(results.keys()):
        impls = impl_names

        # skip only if no impl has metrics
        if all(impl not in results[cfg_id] for impl in impls):
            continue

        meta = config_meta.get(cfg_id, {})
        N = meta.get("N", "NA")
        B = meta.get("B", "NA")
        steps = meta.get("steps", "NA")

        n_impls = len(impls)

        # rows: timers, cols: impls
        values = np.array([
            [
                results[cfg_id].get(impl, {}).get(timer, 0.0)
                for impl in impls
            ]
            for timer in timer_order
        ])

        plt.figure(figsize=(10, 6))
        x = np.arange(n_impls)
        bottoms = np.zeros(n_impls)

        for i, timer in enumerate(timer_order):
            plt.bar(x, values[i], bottom=bottoms, label=timer)
            bottoms += values[i]

        plt.xticks(x, impls, rotation=20)
        plt.ylabel("Time (ms)")
        plt.title(f"Timer Breakdown (stacked, N={N}, B={B}, num_steps={steps})")
        plt.legend()
        plt.tight_layout()
        out_name = f"stacked_timers_{cfg_id}.png".replace("=", "").replace(",", "_")
        plt.savefig(out_name, dpi=200)
        plt.close()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    parse_results(RESULTS_FILE)

    # quick sanity print
    for cfg_id in sorted(results.keys()):
        print(f"\n=== config {cfg_id} ===")
        print("meta:", config_meta.get(cfg_id, {}))
        for impl in impl_names:
            if impl in results[cfg_id]:
                print(f"  {impl}: {results[cfg_id][impl]}")

    plot_e2e_per_config()
    plot_stacked_timers_per_config()

    print("\nGenerated files:")
    print("  e2e_<config>.png")
    print("  stacked_timers_<config>.png")


if __name__ == "__main__":
    main()

