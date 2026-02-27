"""Analyze UNet prediction errors on validation set."""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src_core"))
from model import SegmentationNetwork
from dataloader import yolo_label_to_mask


def infer_patch(model, image, in_channels, device):
    """Run model on a single 224x224 patch, return heatmap."""
    if in_channels == 1:
        t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float() / 255.0
    else:
        t = torch.from_numpy(np.stack([image] * in_channels, axis=0)).unsqueeze(0).float() / 255.0
    t = t.to(device)
    with torch.no_grad():
        out = model(t)
        hm = F.softmax(out, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
    return hm


def main():
    model_path = "checkpoints_ksd2/SODUnet_lr0.001_ep100_bs32_224x224_best.pth"
    images_dir = "KolektorSDD2_patch/mixed/valid/images"
    labels_dir = "KolektorSDD2_patch/mixed/valid/labels"
    output_dir = "error_analysis_unet_mixed"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels", 1)
    model = SegmentationNetwork(in_channels=in_channels, out_channels=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded, in_channels={in_channels}")

    img_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    print(f"Found {len(img_paths)} validation images")

    results = []  # (path, max_score, has_defect, heatmap, image, gt_mask)

    for ip in img_paths:
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        hm = infer_patch(model, img, in_channels, device)

        basename = os.path.splitext(os.path.basename(ip))[0]
        lbl_path = os.path.join(labels_dir, basename + ".txt")
        gt_mask = yolo_label_to_mask(lbl_path, h, w)

        max_score = float(hm.max())
        has_defect = int(np.sum(gt_mask > 0) > 0)
        results.append((ip, max_score, has_defect, hm, img, gt_mask))

    # Sort by score to find optimal threshold
    scores = np.array([r[1] for r in results])
    labels = np.array([r[2] for r in results])

    # Use threshold = 0.5 for classification
    threshold = 0.5
    preds = (scores > threshold).astype(int)

    # Find errors
    fn_indices = [i for i in range(len(results)) if labels[i] == 1 and preds[i] == 0]  # missed defects
    fp_indices = [i for i in range(len(results)) if labels[i] == 0 and preds[i] == 1]  # false alarms

    # Also find borderline cases (defective images with low scores, clean images with high scores)
    defect_scores = [(i, scores[i]) for i in range(len(results)) if labels[i] == 1]
    clean_scores = [(i, scores[i]) for i in range(len(results)) if labels[i] == 0]

    defect_scores.sort(key=lambda x: x[1])  # ascending - lowest score defects first
    clean_scores.sort(key=lambda x: x[1], reverse=True)  # descending - highest score clean first

    print(f"\n=== Error Analysis (threshold={threshold}) ===")
    print(f"Total: {len(results)} images ({sum(labels)} defect, {len(labels) - sum(labels)} clean)")
    print(f"False Negatives (missed defects): {len(fn_indices)}")
    print(f"False Positives (false alarms):   {len(fp_indices)}")
    print(f"Accuracy: {1 - (len(fn_indices) + len(fp_indices)) / len(results):.4f}")

    print(f"\n--- Top 10 hardest defect images (lowest max score) ---")
    for idx, sc in defect_scores[:10]:
        name = os.path.basename(results[idx][0])
        status = "MISS" if sc <= threshold else "ok"
        print(f"  [{status}] {name}: max_score={sc:.4f}")

    print(f"\n--- Top 10 hardest clean images (highest max score) ---")
    for idx, sc in clean_scores[:10]:
        name = os.path.basename(results[idx][0])
        status = "FP" if sc > threshold else "ok"
        print(f"  [{status}] {name}: max_score={sc:.4f}")

    # Save visualizations for error cases + borderline cases
    cases_to_vis = set()
    for i in fn_indices:
        cases_to_vis.add(i)
    for i in fp_indices:
        cases_to_vis.add(i)
    # Also add top-15 borderline from each side
    for idx, _ in defect_scores[:15]:
        cases_to_vis.add(idx)
    for idx, _ in clean_scores[:15]:
        cases_to_vis.add(idx)

    print(f"\nSaving {len(cases_to_vis)} visualizations...")
    for i in cases_to_vis:
        ip, sc, has_def, hm, img, gt = results[i]
        name = os.path.splitext(os.path.basename(ip))[0]

        if has_def and sc <= threshold:
            tag = "FN"
        elif not has_def and sc > threshold:
            tag = "FP"
        elif has_def:
            tag = "TP"
        else:
            tag = "TN"

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=100)

        axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(gt, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(f"GT (defect={has_def})")
        axes[1].axis("off")

        im = axes[2].imshow(hm, cmap="hot", vmin=0, vmax=1)
        axes[2].set_title(f"Heatmap (max={sc:.4f})")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        overlay[hm > 0.5] = [255, 0, 0]
        if gt is not None:
            overlay[gt > 0] = np.clip(
                overlay[gt > 0].astype(int) + np.array([0, 150, 0]), 0, 255
            ).astype(np.uint8)
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay (red=pred, green=GT)")
        axes[3].axis("off")

        fig.suptitle(f"{tag} | {name} | score={sc:.4f}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{tag}_{sc:.4f}_{name}.png"), dpi=100, bbox_inches="tight")
        plt.close()

    print(f"Done! Visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
