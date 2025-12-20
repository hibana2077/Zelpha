"""Visualize Zelpha vs Vanilla results from final_results.csv.

Outputs (by default, saved next to the CSV):
- robust_vs_scale_mean.png: Robust accuracy (Top-1) vs scale factor with mean±std over backbones.
- robust_acc_bar_by_backbone.png: Per-backbone robust accuracy bar chart (Vanilla vs ZELPHA) with Δ.

Optional:
- Use --by-family to also export CNN/ViT family-level scale curves.

Example:
  python docs/results/visualize_results.py --csv docs/results/final_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCALE_POINTS: list[float] = [0.7, 0.85, 1.0, 1.15, 1.3]
SCALE_COLS_ACC1: dict[float, str] = {
    0.7: "scale_0.7_acc1",
    0.85: "scale_0.85_acc1",
    1.0: "scale_1.0_acc1",
    1.15: "scale_1.15_acc1",
    1.3: "scale_1.3_acc1",
}


def _family_from_model_name(model_name: str) -> str:
    name = str(model_name).lower()
    vit_markers = (
        "vit",
        "deit",
        "swin",
        "beit",
        "cait",
        "eva",
        "xcit",
        "pvt",
    )
    if any(m in name for m in vit_markers):
        return "ViT"
    return "CNN"


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in CSV: " + ", ".join(missing)
        )


def _format_percent(y: float) -> str:
    return f"{y:.1f}%"


def plot_robust_vs_scale_mean(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title_suffix: str = "",
) -> None:
    # First average within each backbone (across seeds/runs), then compute mean±std over backbones.
    per_backbone = (
        df.groupby(["model", "Zelpha"], as_index=False)[list(SCALE_COLS_ACC1.values())]
        .mean(numeric_only=True)
        .copy()
    )

    rows = []
    for zelpha_value, label in [("N", "Vanilla"), ("Y", "ZELPHA")]:
        subset = per_backbone[per_backbone["Zelpha"] == zelpha_value]
        means = []
        stds = []
        for s in SCALE_POINTS:
            col = SCALE_COLS_ACC1[s]
            vals = subset[col].astype(float).to_numpy() * 100.0
            means.append(float(np.nanmean(vals)))
            stds.append(float(np.nanstd(vals, ddof=1)) if np.sum(~np.isnan(vals)) >= 2 else 0.0)
        rows.append((label, means, stds))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for label, means, stds in rows:
        ax.errorbar(
            SCALE_POINTS,
            means,
            yerr=stds,
            marker="o",
            linewidth=2.2,
            capsize=4,
            label=f"{label} (mean±std over backbones)",
        )

    ax.set_xlabel("Scale factor s")
    ax.set_ylabel("Accuracy at scale (%)")
    ax.set_xticks(SCALE_POINTS)
    ax.set_xlim(min(SCALE_POINTS) - 0.03, max(SCALE_POINTS) + 0.03)
    ax.set_title("Robust accuracy vs scale factor" + (f" — {title_suffix}" if title_suffix else ""))
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_robust_vs_scale_by_family(df: pd.DataFrame, out_dir: Path) -> None:
    df2 = df.copy()
    df2["family"] = df2["model"].map(_family_from_model_name)

    for fam in ["CNN", "ViT"]:
        subset = df2[df2["family"] == fam]
        if subset.empty:
            continue
        out_path = out_dir / f"robust_vs_scale_{fam.lower()}.png"
        plot_robust_vs_scale_mean(subset, out_path, title_suffix=fam)


def plot_robust_bar_by_backbone(df: pd.DataFrame, out_path: Path) -> None:
    # robust_acc is stored as fraction (0-1) in this CSV.
    per_model = (
        df.groupby(["model", "Zelpha"], as_index=False)["robust_acc"]
        .mean(numeric_only=True)
        .copy()
    )

    pivot = per_model.pivot(index="model", columns="Zelpha", values="robust_acc")
    # Ensure both columns exist; if one is missing, fill with NaN.
    vanilla = pivot.get("N")
    zelpha = pivot.get("Y")
    if vanilla is None:
        pivot["N"] = np.nan
    if zelpha is None:
        pivot["Y"] = np.nan

    pivot = pivot.reset_index().rename(columns={"N": "Vanilla", "Y": "ZELPHA"})
    pivot["Vanilla_pct"] = pivot["Vanilla"].astype(float) * 100.0
    pivot["ZELPHA_pct"] = pivot["ZELPHA"].astype(float) * 100.0
    pivot["delta_pct"] = pivot["ZELPHA_pct"] - pivot["Vanilla_pct"]

    # Count positive gains (ignore NaNs)
    valid_delta = pivot["delta_pct"].dropna()
    n_pos = int((valid_delta > 0).sum())
    n_total = int(valid_delta.shape[0])

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(8.0, 0.65 * len(pivot)), 4.8))

    x = np.arange(len(pivot))
    width = 0.38

    bars1 = ax.bar(x - width / 2, pivot["Vanilla_pct"], width, label="Vanilla")
    bars2 = ax.bar(x + width / 2, pivot["ZELPHA_pct"], width, label="ZELPHA")

    ax.set_ylabel("Robust accuracy (%)")
    ax.set_title(f"Per-backbone robust accuracy (Δ>0: {n_pos}/{n_total})")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot["model"], rotation=35, ha="right")
    ax.legend(frameon=True)

    # Annotate Δ above each group.
    y_max = float(
        np.nanmax(
            np.r_[
                pivot["Vanilla_pct"].to_numpy(dtype=float),
                pivot["ZELPHA_pct"].to_numpy(dtype=float),
            ]
        )
        if len(pivot) else 0.0
    )

    for i, d in enumerate(pivot["delta_pct"].to_numpy(dtype=float)):
        if np.isnan(d):
            continue
        y = max(
            float(pivot.loc[i, "Vanilla_pct"]),
            float(pivot.loc[i, "ZELPHA_pct"]),
        )
        ax.text(
            i,
            y + max(0.6, 0.02 * y_max),
            f"Δ {d:+.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).with_name("final_results.csv"),
        help="Path to final_results.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: same folder as CSV)",
    )
    parser.add_argument(
        "--by-family",
        action="store_true",
        help="Also export CNN/ViT family-level scale curves",
    )
    args = parser.parse_args()

    csv_path: Path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = args.outdir if args.outdir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = ["model", "Zelpha", "robust_acc", *SCALE_COLS_ACC1.values()]
    _require_columns(df, required)

    # Normalize Zelpha column to 'Y'/'N'
    df["Zelpha"] = df["Zelpha"].astype(str).str.strip().str.upper()

    plot_robust_vs_scale_mean(df, out_dir / "robust_vs_scale_mean.png")
    if args.by_family:
        plot_robust_vs_scale_by_family(df, out_dir)

    plot_robust_bar_by_backbone(df, out_dir / "robust_acc_bar_by_backbone.png")

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
