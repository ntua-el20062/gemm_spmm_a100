#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt

csv_path = "memlog_managed_A.csv"
out_png  = "plots/mem_usage_managed_A.png"

t = []
sys_mb = []
gpu_mb = []

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t.append(float(row["t_s"]))
        sys_mb.append(float(row["sys_rss_mb"]))
        gpu_mb.append(float(row["gpu_used_mb"]))

t = np.array(t)
sys_mb = np.array(sys_mb)
gpu_mb = np.array(gpu_mb)

plt.figure(figsize=(7.5, 4.5))

# System memory (red squares)
plt.plot(
    t, sys_mb,
    color="red",
    marker="s",
    markersize=4,
    linewidth=2,
    label="System memory"
)

# GPU memory (yellow triangles)
plt.plot(
    t, gpu_mb,
    color="gold",
    marker="^",
    markersize=4,
    linewidth=2,
    label="GPU memory"
)

plt.xlabel("t (s)")
plt.ylabel("memory (MB)")
plt.title("Memory usage over time (Managed, CPU reading A)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(out_png, dpi=150)
plt.close()

print(f"[OK] Saved {out_png}")

