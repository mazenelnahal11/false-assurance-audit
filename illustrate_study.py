import argparse
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps


INK = "#172d3c"
BLUE = "#006a9f"
ORANGE = "#ad4d22"
GREEN = "#2e7652"
PURPLE = "#79539c"
GRID = "#e1e8ed"
METHODS = (("gradcam", "Grad-CAM"), ("ig", "Integrated Gradients"), ("neutralize", "Neutralization"))
ARCHS = {"mobilenet_v3_large": ("MobileNetV3-Large", BLUE), "resnet34": ("ResNet34", ORANGE)}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
                     "axes.labelcolor": INK, "axes.titleweight": "bold"})


def save(fig, path):
    fig.savefig(path, dpi=350, bbox_inches="tight", pad_inches=.06, facecolor="white")
    plt.close(fig)


def flow_diagram(path):
    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(x, y, w, h, title, details, fill, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.008,rounding_size=.017",
                                   facecolor=fill, edgecolor=edge, linewidth=1.1))
        ax.text(x + .016, y + h - .078, title, weight="bold", fontsize=9, va="top")
        ax.text(x + .016, y + h - .177, details, va="top", fontsize=7.7, linespacing=1.2)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, linewidth=1,
                                    color="#677987"))

    box(.015, .32, .192, .36, "Held-out image", "Same base case\nand checkpoint", "#eef3f6", "#708597")
    box(.275, .56, .370, .39, "Audit evidence", "Three aligned views per policy\nMaximum score + control cutoff", "#e9f4fa", BLUE)
    box(.275, .055, .370, .39, "Behavioral risk label", "Swap marker under blur or occlusion\nCheck prediction flip and margin drop", "#f4edf8", PURPLE)
    box(.733, .30, .251, .40, "Missed-risk accounting", "Count oracle-positive\nand unflagged cases", "#eaf5ed", GREEN)
    for a, b in [((.217, .54), (.265, .74)), ((.217, .46), (.265, .24)),
                 ((.656, .74), (.723, .54)), ((.656, .24), (.723, .46))]:
        arrow(a, b)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save(fig, path)


def operating_points(metrics, path):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharex=True, sharey=True)
    shortcut = metrics[metrics.arm.eq("shortcut")]
    for ax, (method, label) in zip(axes, METHODS, strict=True):
        sub = shortcut[shortcut.method.eq(method)]
        for (arch, seed), pair in sub.groupby(["architecture", "seed"]):
            a, b = (pair[pair.policy.eq(policy)].iloc[0] for policy in ("nominal_only", "context_diverse"))
            x = np.array([a.flag_rate, b.flag_rate]) * 100
            y = np.array([a.missed_risk_rate, b.missed_risk_rate]) * 100
            color = ARCHS[arch][1]
            ax.plot(x, y, color=color, lw=.85, alpha=.67)
            ax.scatter(x[0], y[0], s=28, marker="o", edgecolor=color, facecolor="white", zorder=3)
            ax.scatter(x[1], y[1], s=24, marker="o", color=color, zorder=3)
        ax.set_title(label, loc="left")
        ax.set(xlim=(-2, 103), ylim=(-3, 103), xticks=(0, 25, 50, 75, 100), yticks=(0, 25, 50, 75, 100))
        ax.grid(color=GRID, lw=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Risky cases missed (%)")
    fig.supxlabel("All cases flagged for review (%)", fontsize=8, y=.12)
    handles = [Line2D([0], [0], color=color, lw=1.4, label=name) for name, color in ARCHS.values()]
    handles.extend(Line2D([0], [0], marker="o", color=INK, markerfacecolor=fill, lw=0, label=label)
                   for fill, label in (("white", "Nominal-only"), (INK, "Context-diverse")))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.2, bbox_to_anchor=(.54, -.03))
    fig.tight_layout(rect=[0, .17, 1, 1], w_pad=.7)
    save(fig, path)


def fixed_quota(data, path):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), sharex=True, sharey=True)
    targets = sorted(data.target_review_fraction.unique())
    for ax, (method, name) in zip(axes, METHODS, strict=True):
        selected = data[data.method.eq(method)]
        for (arch, _), cases in selected.groupby(["architecture", "seed"]):
            y = cases.set_index("target_review_fraction").loc[targets, "delta_mrr"] * 100
            ax.plot(range(len(targets)), y, lw=.85, color=ARCHS[arch][1], alpha=.65,
                    marker="o", markersize=2.6)
        means = selected.groupby("target_review_fraction").delta_mrr.mean().loc[targets] * 100
        ax.plot(range(len(targets)), means, lw=1.3, color=INK, marker="D", markersize=3.2, zorder=4)
        ax.axhline(0, color="#6b7f8c", lw=.75)
        ax.axvspan(.83, 1.17, color=GRID, alpha=.75, zorder=0)
        ax.grid(axis="y", color=GRID, lw=.6)
        ax.set_title(name, loc="left")
        ax.set_xticks(range(len(targets)), [f"{x:.0%}" for x in targets])
        ax.set(ylim=(-2.2, 10.5), yticks=(-2, 0, 2, 4, 6, 8, 10))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Missed-risk contrast (pp)")
    fig.supxlabel("Common fraction of cases reviewed", fontsize=8, y=.12)
    handles = [Line2D([0], [0], color=color, lw=1.2, label=name) for name, color in ARCHS.values()]
    handles.append(Line2D([0], [0], color=INK, lw=1.3, marker="D", markersize=3, label="Equal-pair mean"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.2, bbox_to_anchor=(.54, -.02))
    fig.tight_layout(rect=[0, .17, 1, 1], w_pad=.7)
    save(fig, path)


def intervention_example(archive_path, dataset, path):
    with ZipFile(archive_path) as z:
        oracle = pd.read_csv(io.BytesIO(z.read("resnet34_47/oracle_cases.csv")))
        split = pd.read_csv(io.BytesIO(z.read("split_manifest.csv"))).set_index("image_id")
    candidates = oracle[oracle.arm.eq("shortcut") & oracle.y.eq(0)]
    eligible = []
    for image_id, group in candidates.groupby("image_id"):
        g = group.set_index("view")
        if g.aligned_correct.all() and g.loc["nominal", "swapped_correct"] and g.loc[["blur", "occlusion"], "risky"].all():
            eligible.append(image_id)
    assert eligible
    chosen = sorted(eligible)[0]
    with Image.open(dataset / chosen) as img:
        image = ImageOps.fit(ImageOps.exif_transpose(img).convert("RGB"), (112, 112), Image.Resampling.BILINEAR)
    pixels = np.array(image)
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == split.loc[chosen, "sha256"]
    views = {"nominal": pixels, "blur": np.array(image.filter(ImageFilter.GaussianBlur(4))),
             "occlusion": pixels.copy()}
    views["occlusion"][32:80, 32:80] = 128
    report = candidates[candidates.image_id.eq(chosen)].set_index("view")
    fig, axes = plt.subplots(2, 3, figsize=(4.1, 2.8))
    for col, (view, base) in enumerate(views.items()):
        for row, code in enumerate((0, 1)):
            canvas = base.astype(np.float32)
            marker = np.full((20, 20, 1), 20, dtype=np.float32)
            if code == 0:
                marker[:10] = 235
            else:
                marker[:, :10] = 235
            canvas[5:25, 5:25] = .1 * canvas[5:25, 5:25] + .9 * marker
            ax = axes[row, col]
            ax.imshow(canvas / 255)
            if row == 0:
                ax.set_title(view.capitalize(), pad=3)
            correct = bool(report.loc[view, "aligned_correct" if row == 0 else "swapped_correct"])
            ax.set_xlabel("Correct" if correct else "Incorrect", color=GREEN if correct else ORANGE,
                          fontsize=7.5, weight="bold", labelpad=2)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    axes[0, 0].set_ylabel("Aligned", weight="bold")
    axes[1, 0].set_ylabel("Swapped", weight="bold")
    fig.tight_layout(pad=.55, h_pad=.65, w_pad=.15)
    save(fig, path)
    print("Intervention example:", chosen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    flow_diagram(args.output / "audit_workflow.png")
    operating_points(pd.read_csv(args.tables / "verified_metrics.csv"), args.output / "operating_points.png")
    fixed_quota(pd.read_csv(args.tables / "exploratory_fixed_review_quota.csv"), args.output / "review_quota.png")
    if args.archive and args.dataset:
        intervention_example(args.archive, args.dataset, args.output / "intervention_example.png")


if __name__ == "__main__":
    main()
