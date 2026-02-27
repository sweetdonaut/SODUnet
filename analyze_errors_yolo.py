"""Analyze YOLO-seg prediction errors on validation set."""
import numpy as np
import cv2
import os
import sys
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src_core"))
from dataloader import yolo_label_to_mask


def main():
    model_path = "runs/segment/yolo26n-seg-ksd2-mixed-100ep/weights/best.pt"
    images_dir = "KolektorSDD2_patch/mixed/valid/images"
    labels_dir = "KolektorSDD2_patch/mixed/valid/labels"
    output_dir = "error_analysis_yolo_mixed"
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)
    img_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    print(f"Found {len(img_paths)} validation images")

    results_list = []

    for ip in img_paths:
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]

        res = model.predict(ip, imgsz=224, conf=0.01, verbose=False)[0]
        score_map = np.zeros((h, w), dtype=np.float32)

        if res.masks is not None and len(res.masks) > 0:
            masks_data = res.masks.data.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for mask, conf_val in zip(masks_data, confs):
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                score_map = np.maximum(score_map, mask * conf_val)

        basename = os.path.splitext(os.path.basename(ip))[0]
        gt = yolo_label_to_mask(os.path.join(labels_dir, basename + ".txt"), h, w)

        max_score = float(score_map.max())
        has_defect = int(np.sum(gt > 0) > 0)
        results_list.append((ip, max_score, has_defect, score_map, img, gt))

    scores = np.array([r[1] for r in results_list])
    labels = np.array([r[2] for r in results_list])

    # YOLO scores are confidence * mask, use threshold that makes sense
    # For image-level: if max_score > 0 means YOLO detected something
    threshold = 0.01  # same as conf threshold - if YOLO detects anything
    preds = (scores > threshold).astype(int)

    fn_indices = [i for i in range(len(results_list)) if labels[i] == 1 and preds[i] == 0]
    fp_indices = [i for i in range(len(results_list)) if labels[i] == 0 and preds[i] == 1]

    defect_scores = [(i, scores[i]) for i in range(len(results_list)) if labels[i] == 1]
    clean_scores = [(i, scores[i]) for i in range(len(results_list)) if labels[i] == 0]
    defect_scores.sort(key=lambda x: x[1])
    clean_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n=== YOLO Error Analysis (threshold={threshold}) ===")
    print(f"Total: {len(results_list)} images ({sum(labels)} defect, {len(labels) - sum(labels)} clean)")
    print(f"False Negatives (missed defects): {len(fn_indices)}")
    print(f"False Positives (false alarms):   {len(fp_indices)}")

    img_auroc = roc_auc_score(labels, scores)
    print(f"Image AUROC: {img_auroc:.4f}")

    print(f"\n--- Top 10 hardest defect images (lowest max score) ---")
    for idx, sc in defect_scores[:10]:
        name = os.path.basename(results_list[idx][0])
        status = "MISS" if sc <= threshold else "ok"
        print(f"  [{status}] {name}: max_score={sc:.4f}")

    print(f"\n--- Top 10 hardest clean images (highest max score) ---")
    for idx, sc in clean_scores[:10]:
        name = os.path.basename(results_list[idx][0])
        status = "FP" if sc > threshold else "ok"
        print(f"  [{status}] {name}: max_score={sc:.4f}")

    # Visualize errors + borderline cases
    cases_to_vis = set()
    for i in fn_indices:
        cases_to_vis.add(i)
    for i in fp_indices:
        cases_to_vis.add(i)
    for idx, _ in defect_scores[:15]:
        cases_to_vis.add(idx)
    for idx, _ in clean_scores[:15]:
        cases_to_vis.add(idx)

    print(f"\nSaving {len(cases_to_vis)} visualizations...")
    for i in cases_to_vis:
        ip, sc, has_def, hm, img, gt = results_list[i]
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

        h_max = max(hm.max(), 0.01)
        im = axes[2].imshow(hm, cmap="hot", vmin=0, vmax=h_max)
        axes[2].set_title(f"Score map (max={sc:.4f})")
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
