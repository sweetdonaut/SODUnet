"""Evaluate YOLO segmentation model using the same FROC analysis as SODUnet."""
import os
import glob
import argparse
import numpy as np
import cv2
from ultralytics import YOLO
from inference import load_defect_csv, plot_froc_curve, plot_review_efficiency


def extract_yolo_spots(results, filename, orig_h, orig_w):
    """Extract (filename, x, y, score) spots from YOLO segmentation results.

    Computes mask centroids scaled to original image coordinates.
    """
    spots = []
    result = results[0]

    if result.masks is None or len(result.masks) == 0:
        return spots

    masks = result.masks.data.cpu().numpy()  # (N, mask_h, mask_w)
    confs = result.boxes.conf.cpu().numpy()  # (N,)
    mask_h, mask_w = masks.shape[1], masks.shape[2]

    for i in range(len(masks)):
        mask = masks[i]
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            continue

        # Centroid in mask coordinates, scale to original image
        cx = float(np.mean(xs)) * orig_w / mask_w
        cy = float(np.mean(ys)) * orig_h / mask_h
        score = float(confs[i])
        spots.append((filename, cx, cy, score))

    return spots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to best.pt')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Test defects CSV')
    parser.add_argument('--output_dir', type=str, default='../outputs/yolo_eval')
    parser.add_argument('--conf', type=float, default=0.01,
                        help='Confidence threshold (low for full FROC curve)')
    parser.add_argument('--imgsz', type=int, default=1536)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = YOLO(args.model_path)

    # Collect all image paths
    image_paths = sorted(
        glob.glob(os.path.join(args.images_dir, '*.png'))
        + glob.glob(os.path.join(args.images_dir, '*.jpg'))
    )
    num_images = len(image_paths)
    print(f'Found {num_images} test images')

    # Run predictions and collect spots
    all_spots = []
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        orig_h, orig_w = img.shape[:2]

        results = model.predict(
            source=img_path,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
        )
        spots = extract_yolo_spots(results, filename, orig_h, orig_w)
        all_spots.extend(spots)

    # Sort by score descending
    all_spots.sort(key=lambda s: s[3], reverse=True)
    print(f'Total spots: {len(all_spots)}')

    # Load GT
    gt_defects = load_defect_csv(args.csv_path)

    # FROC curves
    print('\nFROC curves:')
    for dsnr_filter, suffix in [(None, 'all'), ('low', 'dsnr_low'), ('high', 'dsnr_high')]:
        out_path = os.path.join(args.output_dir, f'froc_{suffix}.png')
        title = f'FROC Curve ({suffix.replace("_", " ").title()})'
        plot_froc_curve(all_spots, gt_defects, num_images, out_path,
                        title=title, dsnr_filter=dsnr_filter)

    # Review efficiency
    print('\nReview efficiency:')
    for dsnr_filter, suffix in [(None, 'all'), ('low', 'dsnr_low'), ('high', 'dsnr_high')]:
        out_path = os.path.join(args.output_dir, f'review_efficiency_{suffix}.png')
        title = f'Review Efficiency ({suffix.replace("_", " ").title()})'
        plot_review_efficiency(all_spots, gt_defects, out_path,
                               title=title, dsnr_filter=dsnr_filter)


if __name__ == '__main__':
    main()
