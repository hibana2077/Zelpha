import argparse
import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from .dataset import get_dataloaders
from .model import ZelphaModel
from .train import (
    calculate_margin,
    calculate_metrics,
    select_device,
    set_seed,
)


@torch.no_grad()
def collect_features_and_margins(
    model: ZelphaModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_prototype: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect features, logits, labels, and margins for the given model."""

    model.eval()
    all_feats: List[torch.Tensor] = []
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_margins: List[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits, z, dist_sq = model(images)

        margins = calculate_margin(logits, dist_sq, labels, use_prototype=use_prototype)

        all_feats.append(z.cpu())
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_margins.append(margins.cpu())

    feats = torch.cat(all_feats, dim=0).numpy()
    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    margins = torch.cat(all_margins, dim=0).numpy()

    return feats, logits, labels, margins


def tsne_visualization(
    feats: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "t-SNE",
    num_points: int = 2000,
):
    """t-SNE visualization of features."""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n = feats.shape[0]
    if n > num_points:
        idx = np.random.choice(n, num_points, replace=False)
        feats = feats[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, init="pca", random_state=42)
    feats_2d = tsne.fit_transform(feats)

    plt.figure(figsize=(8, 8))
    num_classes = len(np.unique(labels))
    cmap = plt.get_cmap("tab20", num_classes)
    for c in range(num_classes):
        mask = labels == c
        plt.scatter(
            feats_2d[mask, 0],
            feats_2d[mask, 1],
            s=5,
            color=cmap(c),
            label=str(c),
            alpha=0.7,
        )
    plt.legend(fontsize=6, markerscale=3)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def margin_distribution(
    margins_base: np.ndarray,
    margins_zelpha: np.ndarray,
    save_prefix: str,
    bins: int = 100,
):
    """Plot margin histogram and CDF for baseline vs Zelpha."""

    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(
        margins_base,
        bins=bins,
        density=True,
        alpha=0.5,
        label="Baseline",
    )
    plt.hist(
        margins_zelpha,
        bins=bins,
        density=True,
        alpha=0.5,
        label="ZELPHA",
    )
    plt.xlabel("Margin m(x)")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Margin Distribution (Histogram)")
    plt.tight_layout()
    plt.savefig(save_prefix + "_hist.png", dpi=300)
    plt.close()

    # CDF
    def cdf(x: np.ndarray):
        xs = np.sort(x)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        return xs, ys

    x_b, y_b = cdf(margins_base)
    x_z, y_z = cdf(margins_zelpha)

    plt.figure(figsize=(8, 5))
    plt.plot(x_b, y_b, label="Baseline")
    plt.plot(x_z, y_z, label="ZELPHA")
    plt.xlabel("Margin m(x)")
    plt.ylabel("CDF")
    plt.legend()
    plt.title("Margin Distribution (CDF)")
    plt.tight_layout()
    plt.savefig(save_prefix + "_cdf.png", dpi=300)
    plt.close()


@torch.no_grad()
def scale_consistency_visualization(
    model_base: ZelphaModel,
    model_zelpha: ZelphaModel,
    test_loaders: Dict[float, torch.utils.data.DataLoader],
    device: torch.device,
    save_dir: str,
    num_images: int = 8,
):
    """Visualize prediction consistency across scales for representative test images.

    For each chosen image, we save its scaled versions (from loaders) and
    tabulate baseline/ZELPHA predictions per scale.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Collect per-scale predictions and images for a pool of indices
    scales = sorted(test_loaders.keys())

    # Build map: global_index -> {scale: (image_tensor, label, pred_base, pred_zelpha)}
    sample_dict: Dict[int, Dict[float, Tuple[torch.Tensor, int, int, int]]] = {}

    # Assume each loader yields samples in same order across scales.
    for scale in scales:
        loader = test_loaders[scale]
        global_index = 0
        for images, labels in loader:
            b = images.size(0)
            images = images.to(device)
            labels = labels.to(device)

            logits_b, _, _ = model_base(images)
            logits_z, _, _ = model_zelpha(images)

            preds_b = logits_b.argmax(dim=1)
            preds_z = logits_z.argmax(dim=1)

            for i in range(b):
                gi = global_index + i
                if gi not in sample_dict:
                    sample_dict[gi] = {}
                sample_dict[gi][scale] = (
                    images[i].cpu(),
                    int(labels[i].cpu().item()),
                    int(preds_b[i].cpu().item()),
                    int(preds_z[i].cpu().item()),
                )

            global_index += b

    # Select representative images:
    # ZELPHA correct at main scale (assume 1.0) and baseline wrong.
    main_scale = min(scales, key=lambda s: abs(s - 1.0))
    candidates: List[int] = []

    for gi, per_scale in sample_dict.items():
        if main_scale not in per_scale:
            continue
        _, label, pred_b, pred_z = per_scale[main_scale]
        if (pred_z == label) and (pred_b != label):
            candidates.append(gi)

    if len(candidates) == 0:
        # Fallback: just take first indices
        candidates = sorted(sample_dict.keys())

    random.shuffle(candidates)
    chosen = candidates[:num_images]

    # For each chosen index, save scaled images in a row and write a txt for labels.
    info_lines: List[str] = []

    for idx, gi in enumerate(chosen):
        per_scale = sample_dict[gi]
        cols = len(scales)

        # Figure for images
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
        if cols == 1:
            axes = [axes]

        row_info = [f"Image {idx} (global_idx={gi})"]

        for j, scale in enumerate(scales):
            if scale not in per_scale:
                continue
            img_t, label, pred_b, pred_z = per_scale[scale]

            img_np = img_t.numpy()
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            # Simple [0,1] clipping if values are already normalized.
            img_np = np.clip(img_np, 0.0, 1.0)

            ax = axes[j]
            ax.imshow(img_np)
            ax.axis("off")
            ax.set_title(f"scale={scale}")

            row_info.append(
                f"  scale={scale}: label={label}, base={pred_b}, zelpha={pred_z}"
            )

        fig.tight_layout()
        fig_path = os.path.join(save_dir, f"scale_consistency_{idx}.png")
        plt.savefig(fig_path, dpi=200)
        plt.close(fig)

        info_lines.append("\n".join(row_info))

    # Write prediction info
    info_path = os.path.join(save_dir, "scale_consistency_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(info_lines))


def main():
    parser = argparse.ArgumentParser(description="Visualization utilities for ZELPHA vs baseline")

    # Checkpoints
    parser.add_argument("--ckpt_zelpha", type=str, required=True, help="ZELPHA checkpoint path")
    parser.add_argument("--ckpt_base", type=str, required=True, help="Baseline checkpoint path")

    # Dataset / model config (must match training)
    parser.add_argument("--dataset", type=str, default="UC_Merced")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--model_name", type=str, default="zelpha")
    parser.add_argument("--no_lipschitz", action="store_true")
    parser.add_argument("--no_scale_pooling", action="store_true")
    parser.add_argument("--num_prototypes", type=int, default=3)

    # Tasks / output
    parser.add_argument("--do_tsne", action="store_true")
    parser.add_argument("--do_margin", action="store_true")
    parser.add_argument("--do_scale_vis", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save visualizations")

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    train_loader, val_loader, test_loaders, num_classes = get_dataloaders(
        batch_size=args.batch_size,
        image_size=args.image_size,
        dataset_name=args.dataset,
    )

    # Build baseline (linear head) model
    # NOTE: baseline 定義為「沒有 Lipschitz、沒有 multi-scale pooling」
    model_base = ZelphaModel(
        num_classes=num_classes,
        model_name=args.model_name,
        use_spectral_norm=False,
        use_prototype=False,
        use_scale_pooling=False,
        num_prototypes=args.num_prototypes,
    ).to(device)

    # Build ZELPHA (prototype) model
    # NOTE: ZELPHA 依照參數決定是否使用 Lipschitz / multi-scale pooling
    model_zelpha = ZelphaModel(
        num_classes=num_classes,
        model_name=args.model_name,
        use_spectral_norm=not args.no_lipschitz,
        use_prototype=True,
        use_scale_pooling=not args.no_scale_pooling,
        num_prototypes=args.num_prototypes,
    ).to(device)

    # Load checkpoints (state_dict expected)
    state_base = torch.load(args.ckpt_base, map_location=device)
    state_zelpha = torch.load(args.ckpt_zelpha, map_location=device)

    if isinstance(state_base, dict) and "state_dict" in state_base:
        state_base = state_base["state_dict"]
    if isinstance(state_zelpha, dict) and "state_dict" in state_zelpha:
        state_zelpha = state_zelpha["state_dict"]

    model_base.load_state_dict(state_base, strict=False)
    model_zelpha.load_state_dict(state_zelpha, strict=False)

    model_base.eval()
    model_zelpha.eval()

    # Select a reference test loader (e.g., the scale closest to 1.0)
    scales = sorted(test_loaders.keys())
    ref_scale = min(scales, key=lambda s: abs(s - 1.0))
    ref_loader = test_loaders[ref_scale]

    # Collect data for reference scale
    feats_base, logits_base, labels, margins_base = collect_features_and_margins(
        model_base, ref_loader, device, use_prototype=False
    )
    feats_zelpha, logits_zelpha, labels2, margins_zelpha = collect_features_and_margins(
        model_zelpha, ref_loader, device, use_prototype=True
    )

    assert np.all(labels == labels2), "Label mismatch between baseline and ZELPHA on ref loader"

    # Optionally restrict to cases where ZELPHA correct & baseline wrong, or rank by loss
    with torch.no_grad():
        labels_t = torch.from_numpy(labels)
        logits_base_t = torch.from_numpy(logits_base)
        logits_zelpha_t = torch.from_numpy(logits_zelpha)

        preds_base = logits_base_t.argmax(dim=1)
        preds_zelpha = logits_zelpha_t.argmax(dim=1)

        acc1_base, _, _ = calculate_metrics(logits_base_t, labels_t, num_classes=num_classes)
        acc1_zelpha, _, _ = calculate_metrics(logits_zelpha_t, labels_t, num_classes=num_classes)
        print(f"Ref scale Acc@1: baseline={acc1_base:.4f}, ZELPHA={acc1_zelpha:.4f}")

        mask_good_zelpha = (preds_zelpha == labels_t) & (preds_base != labels_t)

        if mask_good_zelpha.sum().item() < 10:
            # Fallback: sort by CE loss improvement
            ce = torch.nn.CrossEntropyLoss(reduction="none")
            loss_base = ce(logits_base_t, labels_t)
            loss_zelpha = ce(logits_zelpha_t, labels_t)
            diff = loss_base - loss_zelpha
            # pick top-k
            k = min(500, diff.numel())
            topk_idx = torch.topk(diff, k=k).indices
            mask_good_zelpha = torch.zeros_like(diff, dtype=torch.bool)
            mask_good_zelpha[topk_idx] = True

        mask_np = mask_good_zelpha.cpu().numpy().astype(bool)

        feats_base_sel = feats_base[mask_np]
        feats_zelpha_sel = feats_zelpha[mask_np]
        labels_sel = labels[mask_np]
        margins_base_sel = margins_base[mask_np]
        margins_zelpha_sel = margins_zelpha[mask_np]

    # 1) t-SNE
    if args.do_tsne:
        tsne_visualization(
            feats_base_sel,
            labels_sel,
            save_path=os.path.join(args.output_dir, "tsne_baseline.png"),
            title="Baseline t-SNE (selected)",
        )
        tsne_visualization(
            feats_zelpha_sel,
            labels_sel,
            save_path=os.path.join(args.output_dir, "tsne_zelpha.png"),
            title="ZELPHA t-SNE (selected)",
        )

    # 2) Margin distribution (selected set)
    if args.do_margin:
        margin_distribution(
            margins_base_sel,
            margins_zelpha_sel,
            save_prefix=os.path.join(args.output_dir, "margin_selected"),
        )

    # 3) Scale consistency visualization
    if args.do_scale_vis:
        scale_consistency_visualization(
            model_base,
            model_zelpha,
            test_loaders,
            device,
            save_dir=os.path.join(args.output_dir, "scale_vis"),
        )


if __name__ == "__main__":
    main()
