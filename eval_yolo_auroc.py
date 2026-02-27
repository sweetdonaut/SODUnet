"""Evaluate YOLO-seg model with Image/Pixel AUROC — same metrics as UNet."""
import torch
import numpy as np
import cv2
import os
import glob
import argparse
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO

# reuse the same mask rasterizer as UNet
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src_core'))
from dataloader import yolo_label_to_mask


def evaluate_yolo_auroc(model_path, images_dir, labels_dir, imgsz=512, conf=0.01):
    model = YOLO(model_path)

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png"))
        + glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.PNG"))
        + glob.glob(os.path.join(images_dir, "*.JPG"))
    )
    print(f"Found {len(image_paths)} images")

    all_scores = []
    all_labels = []
    pixel_scores = []
    pixel_labels = []

    for img_path in image_paths:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        h, w = image.shape[:2]

        # YOLO inference
        results = model.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)
        result = results[0]

        # Build pixel-level score map from YOLO masks
        score_map = np.zeros((h, w), dtype=np.float32)

        if result.masks is not None and len(result.masks) > 0:
            masks_data = result.masks.data.cpu().numpy()  # (N, mask_h, mask_w)
            confs = result.boxes.conf.cpu().numpy()        # (N,)

            for mask, conf_val in zip(masks_data, confs):
                # Resize mask to original image size
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                # Confidence-weighted overlay (take max at each pixel)
                score_map = np.maximum(score_map, mask * conf_val)

        max_score = score_map.max()

        # Load GT mask from YOLO label
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, basename + ".txt")
        gt_mask = yolo_label_to_mask(label_path, h, w).astype(np.float32) / 255.0

        has_anomaly = 1.0 if np.sum(gt_mask) > 0 else 0.0
        all_scores.append(max_score)
        all_labels.append(has_anomaly)

        pixel_scores.extend(score_map.flatten())
        pixel_labels.extend(gt_mask.flatten())

    # Calculate AUROC
    image_auroc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    pixel_auroc = roc_auc_score(pixel_labels, pixel_scores) if len(np.unique(pixel_labels)) > 1 else 0.0

    return image_auroc, pixel_auroc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='YOLO .pt path')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--labels_dir', type=str, required=True)
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--conf', type=float, default=0.01,
                        help='Low conf threshold to get soft scores')
    args = parser.parse_args()

    image_auroc, pixel_auroc = evaluate_yolo_auroc(
        args.model, args.images_dir, args.labels_dir, args.imgsz, args.conf
    )
    print(f"\nYOLO Results:")
    print(f"  Image AUROC: {image_auroc:.4f}")
    print(f"  Pixel AUROC: {pixel_auroc:.4f}")


if __name__ == "__main__":
    main()
