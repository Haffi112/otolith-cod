#!/usr/bin/env python3
"""Figures for the manuscript appendix, all read from archived results.

Three figures, none of which needs a GPU or any recomputation:

  forward_selection_comparison.png     Macro F1 as metadata groups are added
                                       greedily to the embedding, frozen against
                                       adapted. The visual form of the forward
                                       selection numbers quoted in the text.
  permutation_importance_comparison.png F1 drop when each feature group is
                                       shuffled, frozen against adapted. Shows
                                       the morphometric variables going from
                                       costly to shuffle to free to shuffle.
  encoder_comparison.png               All four encoders, frozen and adapted.
                                       The adapted bars are seed means with a
                                       seed standard deviation, which is the
                                       same summary the encoder table uses.
                                       The earlier version of this figure drew
                                       the seed-42 run with its bootstrap
                                       interval, so figure and table disagreed
                                       on both the value and the error bar.

Inputs are the feature_importance.json files of the attribution batch and the
bootstrap results.json of every full-training-set adapted run under results/.
Runs that appear in several batches are identical (extraction is deterministic
on one accelerator) and are counted once.

    uv run python scripts/make_appendix_figures.py
"""

import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
FIGURES = Path("outputs/paper_figures")
ATTRIBUTION = RESULTS / "attribution" / "results"
FROZEN_COLOR, LORA_COLOR = "#4C72B0", "#C44E52"
ENCODERS = [("clip", "CLIP"), ("siglip2", "SigLIP2"), ("dinov2", "DINOv2"), ("dinov3", "DINOv3")]
PRETTY = {"SigLIP Embeddings": "Embedding", "Length": "Fish length", "Month": "Month",
          "Latitude": "Latitude", "Longitude": "Longitude", "Is Survey": "Survey flag",
          "Seg Width": "Otolith width", "Seg Height": "Otolith height",
          "Seg Aspect Ratio": "Aspect ratio"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})


def importance(condition):
    path = ATTRIBUTION / condition / "bootstrap" / "feature_importance" / "feature_importance.json"
    return json.loads(path.read_text())


def figure_forward_selection(frozen, lora, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for data, label, color, marker in [(frozen, "Frozen", FROZEN_COLOR, "o"),
                                       (lora, "LoRA-adapted", LORA_COLOR, "s")]:
        steps = data["forward_selection"]
        f1 = [100 * s["f1_mean"] for s in steps]
        ax.plot(range(len(steps)), f1, marker=marker, color=color, label=label, lw=1.6)
        for i, s in enumerate(steps):
            name = PRETTY[s["feature_added"]]
            offset = (0, -11) if data is frozen else (0, 7)
            ax.annotate(name, (i, f1[i]), textcoords="offset points", xytext=offset,
                        ha="center", fontsize=6.5, color=color)
    ax.set_xlabel("Metadata groups added, in order of attribution")
    ax.set_ylabel("Macro F1 (%)")
    ax.set_xticks(range(9))
    ax.set_xticklabels(["embedding\nalone"] + [f"+{i}" for i in range(1, 9)])
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def figure_permutation(frozen, lora, path):
    names = list(frozen["permutation_importances"][0].keys())
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for offset, data, label, color in [(-0.2, frozen, "Frozen", FROZEN_COLOR),
                                       (0.2, lora, "LoRA-adapted", LORA_COLOR)]:
        imp = data["permutation_importances"][0]
        means = [100 * imp[n]["mean"] for n in names]
        stds = [100 * imp[n]["std"] for n in names]
        ax.bar(x + offset, means, 0.4, yerr=stds, capsize=2.5, color=color, label=label)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[n] for n in names], rotation=30, ha="right")
    ax.set_ylabel("Macro F1 drop when shuffled (points)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def full_set_runs():
    """Every adapted run at the full training set, once per run name."""
    runs = {}
    for p in RESULTS.glob("**/bootstrap/results.json"):
        m = re.search(r"/([a-z0-9]+)_lora_r16a32_s(\d+)_clahe/bootstrap/results.json", str(p))
        if m:
            agg = json.loads(p.read_text())["aggregated_results"]
            runs[(m.group(1), int(m.group(2)))] = (agg["accuracy"]["mean"], agg["f1"]["mean"])
    return runs


def frozen_results():
    out = {}
    for encoder, _ in ENCODERS:
        hits = sorted(RESULTS.glob(f"**/{encoder}-frozen/bootstrap/results.json"))
        agg = json.loads(hits[0].read_text())["aggregated_results"]
        out[encoder] = {k: (agg[k]["mean"], agg[k]["ci_lower"], agg[k]["ci_upper"])
                        for k in ("accuracy", "f1")}
    return out


def figure_encoders(path):
    runs, frozen = full_set_runs(), frozen_results()
    x = np.arange(len(ENCODERS))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, idx, title in [(axes[0], "accuracy", 0, "Accuracy"), (axes[1], "f1", 1, "Macro F1")]:
        fz = [frozen[e][key] for e, _ in ENCODERS]
        ax.bar(x - 0.2, [v[0] for v in fz], 0.4, yerr=[[v[0] - v[1] for v in fz], [v[2] - v[0] for v in fz]],
               capsize=3, color=FROZEN_COLOR, label="Frozen (95% bootstrap interval)")
        means, sds, ns = [], [], []
        for e, _ in ENCODERS:
            vals = [v[idx] for (enc, seed), v in runs.items() if enc == e]
            means.append(np.mean(vals)); sds.append(np.std(vals, ddof=1)); ns.append(len(vals))
        ax.bar(x + 0.2, means, 0.4, yerr=sds, capsize=3, color=LORA_COLOR,
               label="LoRA-adapted (seed mean and sd)")
        for xi, n in zip(x, ns):
            ax.text(xi + 0.2, 0.455, f"n={n}", ha="center", fontsize=7, color="white")
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in ENCODERS])
        ax.set_ylabel(title); ax.set_title(title); ax.set_ylim(0.45, 0.75)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        print(title, {l: (round(m, 4), round(s, 4), n) for (_, l), m, s, n in zip(ENCODERS, means, sds, ns)})
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    frozen, lora = importance("siglip2-frozen"), importance("siglip2_lora_r16a32_s42_clahe")
    figure_forward_selection(frozen, lora, FIGURES / "forward_selection_comparison.png")
    figure_permutation(frozen, lora, FIGURES / "permutation_importance_comparison.png")
    figure_encoders(FIGURES / "encoder_comparison.png")
    print("written to", FIGURES)


if __name__ == "__main__":
    main()
