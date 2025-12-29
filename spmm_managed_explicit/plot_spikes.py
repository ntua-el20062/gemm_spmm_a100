#!/usr/bin/env python3
import re
import sys
import os
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} results.txt output.png")
    sys.exit(1)

input_txt = sys.argv[1]
output_png = sys.argv[2]

# ---- Title from filename ----
base = os.path.basename(input_txt)
title = os.path.splitext(base)[0]
title = title.replace("_", " ")

# ---- Regex for iteration lines ----
pattern = re.compile(
    r"iter:\s*(\d+),\s*t_iter:\s*([0-9.]+)\s*ms"
)

iters = []
times = []

with open(input_txt, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            iters.append(int(m.group(1)))
            times.append(float(m.group(2)))

if not iters:
    raise RuntimeError("No iteration data found in input file")

# ---- Plot ----
plt.figure(figsize=(10, 4))
plt.plot(iters, times, linewidth=1)

spikes = [0, 250, 500, 750, 1000]
plt.xticks(spikes)

plt.xlabel("Iteration")
plt.ylabel("Time (ms)")
plt.title(title)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_png, dpi=150)
print(f"Saved plot to {output_png}")

