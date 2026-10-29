"""Plot the CNN POC curves: train loss and held-out balanced accuracy vs samples seen, small and full tier,
plain shuffle vs class-balanced sampling. usage: python poc_plot.py <poc dir> [max_lr]"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

d = Path(sys.argv[1])
df = pd.concat([pd.read_csv(p) for p in sorted(d.glob("*_losses.csv")) if p.name != "poc_losses_all.csv"])
df.to_csv(d / "poc_losses_all.csv", index=False)
if len(sys.argv) > 2:
    df = df[df.max_lr == float(sys.argv[2])]
colors = {("small", "shuffle"): "tab:blue", ("small", "balanced"): "tab:cyan",
          ("full", "shuffle"): "tab:red", ("full", "balanced"): "tab:orange"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for (tier, sampler), g in df.groupby(["tier", "sampler"]):
    c, lab = colors.get((tier, sampler), "k"), f"{tier}, {sampler}"
    tr = g[g.metric == "train_loss"]
    axes[0].plot(tr.samples_seen, tr.value, color=c, lw=1.2, label=lab)
    va = g[g.metric == "val_balanced_acc"]
    axes[1].plot(va.samples_seen, va.value, color=c, marker="o", lw=2, label=lab)
axes[0].set_ylabel("train cross-entropy (every 10 batches)")
axes[1].set_ylabel("held-out balanced accuracy (per epoch)")
for ax in axes:
    ax.set_xlabel("training images seen")
    ax.legend(frameon=False)
fig.suptitle("Tiny CNN on 48 px crops streamed from disk by annbatch (held out: 6 of 64 plates)")
fig.tight_layout()
fig.savefig(d / "poc_losses.png", dpi=150)
print(d / "poc_losses.png")
