import re
import numpy as np
import matplotlib.pyplot as plt
LOG_FILE = "results_all_impl_compared.txt"   # change if your filename is different

with open(LOG_FILE, "r") as f:
    lines = f.readlines()

data = {}

current_impl = None    # "initial", "double", "full"
current_cfg = None     # (N, batch, steps)

for raw in lines:
    line = raw.strip()

    # detect which implementation block we're in
    if "INITIAL APPROACH" in line:
        current_impl = "initial"
        current_cfg = None
        continue
    if "DOUBLE BUFFERING" in line:
        current_impl = "double"
        current_cfg = None
        continue
    if "FULL OVERLAP" in line:
        current_impl = "full"
        current_cfg = None
        continue

    # detect configuration line
    if line.startswith("Using N ="):
        m = re.search(
            r"Using N\s*=\s*(\d+),\s*(?:batch_size|buff_size)\s*=\s*(\d+)\s*\(num_steps\s*=\s*(\d+)\)",
            line,
        )
        if m:
            N = int(m.group(1))
            batch_or_buff = int(m.group(2))
            steps = int(m.group(3))
            current_cfg = (N, batch_or_buff, steps)
            data.setdefault(current_cfg, {})
        continue

    # mark failures (CUDA error or killed process)
    if "CUDA error" in line or "Killed" in line:
        if current_impl is not None and current_cfg is not None:
            data.setdefault(current_cfg, {})
            data[current_cfg][current_impl] = None
        current_impl = None
        current_cfg = None
        continue

    # pick up t_end_2_end
    if "t_end_2_end" in line and current_impl is not None and current_cfg is not None:
        m = re.search(r"t_end_2_end\s*=\s*([0-9.]+)", line)
        if m:
            t = float(m.group(1))
            data[current_cfg][current_impl] = t

valid_Ns = {20000, 25000, 30000}  
valid_size = {5, 10}
valid_steps = {60, 80, 100, 150}
valid_cfgs = []

for cfg, impls in data.items():
    N, batch, steps = cfg
    if N not in valid_Ns:
        continue
    if batch not in valid_size:
        continue
    if steps not in valid_steps:
        continue
    if all(
        impls.get(name) is not None and isinstance(impls.get(name), float)
        for name in ("initial", "double", "full")
    ):
        valid_cfgs.append(cfg)

valid_cfgs.sort()


if not valid_cfgs:
    raise RuntimeError("No configurations where all three implementations finished for N=20/25/30k")

labels = []
t_initial = []
t_double = []
t_full = []

for N, batch, steps in valid_cfgs:
    labels.append(f"N={N//1000}k, b={batch}, s={steps}")
    impls = data[(N, batch, steps)]
    t_initial.append(impls["initial"])
    t_double.append(impls["double"])
    t_full.append(impls["full"])

x = np.arange(len(valid_cfgs))
width = 0.25

plt.figure(figsize=(max(10, len(valid_cfgs) * 0.6), 6))

plt.bar(x - width, t_initial, width, label="Initial approach")
plt.bar(x,          t_double,  width, label="Double buffering")
plt.bar(x + width,  t_full,    width, label="Full overlap")

plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("End-to-end time (ms)")
plt.title("End-to-end time (t_end_2_end) for N = 20k, 25k, 30k\nOnly configs where all implementations completed")
plt.legend()
plt.tight_layout()
plt.grid(axis="y", alpha=0.3)

plt.savefig("end2end.png", dpi=300, bbox_inches="tight")

