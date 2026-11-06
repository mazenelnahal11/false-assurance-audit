from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
tables = root / "verified_study"
out = root / "paper" if (root / "paper").is_dir() else root
out.mkdir(exist_ok=True)
effects = pd.read_csv(tables / "verified_paired_contrasts.csv")
metrics = pd.read_csv(tables / "verified_metrics.csv")
styles = {"gradcam": ("Grad-CAM", "#00689d"), "ig": ("Integrated Gradients", "#b14b17"),
          "neutralize": ("Neutralization", "#337a53")}
labels = [f"{arch} {seed}" for arch in ("MobileNet", "ResNet") for seed in (11, 29, 47)]
positions = np.arange(6)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
for index, (method, (label, color)) in enumerate(styles.items()):
    data = effects[effects.method == method].sort_values(["architecture", "seed"])
    y = positions + (1 - index) * .20
    axes[0].errorbar(data.delta_mrr * 100, y,
                     xerr=np.array([data.delta_mrr - data.bootstrap_low, data.bootstrap_high - data.delta_mrr]) * 100,
                     fmt="o", markersize=3.2, lw=1, color=color, label=label, capsize=1.8)
    flags = metrics[(metrics.arm == "shortcut") & (metrics.method == method)].pivot(
        index=["architecture", "seed"], columns="policy", values="flag_rate")
    axes[1].scatter((flags.context_diverse - flags.nominal_only) * 100, y, s=14, color=color)
for ax, title, xlabel in zip(
        axes, ("A. Missed-risk contrast", "B. Change in cases flagged"),
        ("Nominal minus diverse (percentage points)", "Diverse minus nominal (percentage points)"),
        strict=True):
    ax.axvline(0, lw=.8, color="#313d45")
    ax.axhline(2.5, lw=.7, color="#aab4bb")
    ax.set_yticks(positions, labels if ax is axes[0] else [])
    ax.set_ylim(-.6, 5.6)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="both", labelsize=8, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, color="#e2e7ea", lw=.5)
    ax.set_axisbelow(True)
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, ncol=3, fontsize=8, loc="lower center", frameon=False, bbox_to_anchor=(.54, -.02))
fig.tight_layout(rect=[0, .10, 1, 1], w_pad=1)
fig.savefig(out / "paired_effects.png", dpi=300, bbox_inches="tight", pad_inches=.07)
plt.close(fig)
