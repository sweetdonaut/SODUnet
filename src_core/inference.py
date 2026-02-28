import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import argparse
import glob
import csv
import importlib
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataloader import calculate_positions, yolo_label_to_mask


class InferenceDataset(Dataset):

    def __init__(self, images_dir):
        self.image_paths = sorted(
            glob.glob(os.path.join(images_dir, "*.png"))
            + glob.glob(os.path.join(images_dir, "*.jpg"))
            + glob.glob(os.path.join(images_dir, "*.PNG"))
            + glob.glob(os.path.join(images_dir, "*.JPG"))
            + glob.glob(os.path.join(images_dir, "*.tiff"))
            + glob.glob(os.path.join(images_dir, "*.tif"))
            + glob.glob(os.path.join(images_dir, "*.TIFF"))
            + glob.glob(os.path.join(images_dir, "*.TIF"))
        )
        print(f"Found {len(self.image_paths)} test images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        image = image.astype(np.float32)
        original_h, original_w = image.shape[:2]

        return {
            'image': image,
            'image_path': img_path,
            'original_size': (original_h, original_w)
        }


def sliding_window_inference(image, model, patch_size, in_channels, device):
    h, w = image.shape[:2]

    if h < patch_size or w < patch_size:
        raise ValueError(f"Image size ({h}x{w}) is smaller than patch size ({patch_size}x{patch_size}).")

    y_positions = calculate_positions(h, patch_size)
    x_positions = calculate_positions(w, patch_size)

    if y_positions is None or x_positions is None:
        raise ValueError(f"Image size ({h}x{w}) is too small for patch size ({patch_size}x{patch_size})")

    output_heatmap = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    for y_idx, y in enumerate(y_positions):
        for x_idx, x in enumerate(x_positions):
            patch = image[y:y+patch_size, x:x+patch_size]

            if in_channels == 1:
                input_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float() / 255.0
            else:
                patch_ch = np.stack([patch] * in_channels, axis=0)
                input_tensor = torch.from_numpy(patch_ch).unsqueeze(0).float() / 255.0
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                output = model(input_tensor)
                output_sm = F.softmax(output, dim=1)
                patch_heatmap = output_sm[:, 1, :, :].squeeze().cpu().numpy()

            if len(y_positions) > 1 or len(x_positions) > 1:
                # Center-crop stitching: each patch only contributes its non-overlapping center
                y_stride = y_positions[1] - y_positions[0] if len(y_positions) > 1 else patch_size
                y_margin = (patch_size - y_stride) // 2

                if y_idx == 0:
                    y_start_crop, y_end_crop = 0, patch_size - y_margin
                elif y_idx == len(y_positions) - 1:
                    y_start_crop, y_end_crop = y_margin, patch_size
                else:
                    y_start_crop, y_end_crop = y_margin, patch_size - y_margin

                if len(x_positions) > 1:
                    x_stride = x_positions[1] - x_positions[0]
                    x_margin = (patch_size - x_stride) // 2

                    if x_idx == 0:
                        x_start_crop, x_end_crop = 0, patch_size - x_margin
                    elif x_idx == len(x_positions) - 1:
                        x_start_crop, x_end_crop = x_margin, patch_size
                    else:
                        x_start_crop, x_end_crop = x_margin, patch_size - x_margin
                else:
                    x_start_crop, x_end_crop = 0, patch_size

                patch_region = patch_heatmap[y_start_crop:y_end_crop, x_start_crop:x_end_crop]

                oy_s, oy_e = y + y_start_crop, y + y_end_crop
                ox_s, ox_e = x + x_start_crop, x + x_end_crop

                output_heatmap[oy_s:oy_e, ox_s:ox_e] = patch_region
                weight_map[oy_s:oy_e, ox_s:ox_e] = 1
            else:
                output_heatmap[y:y+patch_size, x:x+patch_size] = patch_heatmap
                weight_map[y:y+patch_size, x:x+patch_size] = 1

    output_heatmap = output_heatmap / np.maximum(weight_map, 1)
    return output_heatmap, image


def visualize_results(image, heatmap, output_path, gt_mask=None):
    ncols = 3 if gt_mask is None else 4
    fig = plt.figure(figsize=(4 * ncols, 4), dpi=200)
    gs = gridspec.GridSpec(1, ncols, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Image')
    ax1.axis('off')

    col = 1
    if gt_mask is not None:
        ax_gt = fig.add_subplot(gs[col])
        ax_gt.imshow(gt_mask, cmap='gray', vmin=0, vmax=255)
        ax_gt.set_title('GT Mask')
        ax_gt.axis('off')
        col += 1

    ax_hm = fig.add_subplot(gs[col])
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min < 1e-8:
        h_min, h_max = 0, 1
    im = ax_hm.imshow(heatmap, cmap='hot', vmin=h_min, vmax=h_max)
    ax_hm.set_title('Heatmap')
    ax_hm.axis('off')
    plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    col += 1

    # Overlay: red prediction on gray image
    pred_bin = (heatmap > 0.5).astype(np.uint8) * 255
    overlay = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    overlay[pred_bin > 0] = [255, 0, 0]
    ax_ov = fig.add_subplot(gs[col])
    ax_ov.imshow(overlay)
    ax_ov.set_title('Overlay')
    ax_ov.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def extract_spots(heatmap, threshold=0.1):
    """Extract bright spots from heatmap via connected components.
    Returns list of (x, y, score) sorted by score descending.
    """
    binary = (heatmap > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    spots = []
    for label_id in range(1, num_labels):  # skip background
        component_mask = (labels == label_id)
        masked_scores = heatmap * component_mask
        peak_idx = np.unravel_index(masked_scores.argmax(), masked_scores.shape)
        peak_y, peak_x = peak_idx
        peak_score = float(heatmap[peak_y, peak_x])
        spots.append((peak_x, peak_y, peak_score))
    spots.sort(key=lambda s: s[2], reverse=True)
    return spots


def load_defect_csv(csv_path):
    """Load test defect CSV. Returns dict: filename -> list of (rawx, rawy, dsnr)."""
    gt = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row['filename']
            gt.setdefault(fn, []).append((
                int(row['rawx']), int(row['rawy']), float(row['dsnr'])
            ))
    return gt


def plot_review_efficiency(all_spots, gt_defects, output_path, title='Review Efficiency',
                           dsnr_filter=None):
    """Plot review efficiency: x=review count (sorted by score desc), y=captured defects.

    all_spots: list of (filename, x, y, score) already sorted by score descending.
    gt_defects: dict filename -> list of (rawx, rawy, dsnr).
    dsnr_filter: None=all, 'low'=dsnr<3.5, 'high'=dsnr>=3.5.
    """
    # Filter GT defects by DSNR
    if dsnr_filter == 'low':
        filtered_gt = {fn: [(x, y, d) for x, y, d in defs if d < 3.5]
                       for fn, defs in gt_defects.items()}
    elif dsnr_filter == 'high':
        filtered_gt = {fn: [(x, y, d) for x, y, d in defs if d >= 3.5]
                       for fn, defs in gt_defects.items()}
    else:
        filtered_gt = gt_defects
    # Remove empty entries
    filtered_gt = {fn: defs for fn, defs in filtered_gt.items() if len(defs) > 0}
    total_defects = sum(len(v) for v in filtered_gt.values())

    if total_defects == 0:
        print(f"  {title}: no defects in this group, skipping")
        return

    captured = {fn: [False] * len(defs) for fn, defs in filtered_gt.items()}
    review_counts = []
    defect_counts = []
    cum_defects = 0

    for i, (fn, sx, sy, score) in enumerate(all_spots):
        if fn in captured:
            for j, (gx, gy, _dsnr) in enumerate(filtered_gt[fn]):
                if not captured[fn][j] and abs(sx - gx) < 15 and abs(sy - gy) < 15:
                    captured[fn][j] = True
                    cum_defects += 1
                    break
        review_counts.append(i + 1)
        defect_counts.append(cum_defects)

    # Plot (prepend origin so curve starts at (0, 0))
    review_counts = [0] + review_counts
    defect_counts = [0] + defect_counts
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(review_counts, defect_counts, 'b-', linewidth=2)
    ax.set_xlabel('Review Count (spots sorted by score, high→low)')
    ax.set_ylabel('Captured Defect Count')
    ax.set_title(title)
    ax.axhline(y=total_defects, color='r', linestyle='--', alpha=0.7,
               label=f'Total defects: {total_defects}')
    if len(review_counts) > 0:
        ax.set_xlim(0, review_counts[-1])
    ax.set_ylim(0, total_defects + 1)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  {title}: {cum_defects}/{total_defects} defects captured, "
          f"{len(all_spots)} spots -> {output_path}")


def inference(args):
    output_dir = os.path.join('..', 'outputs', args.task_name)
    os.makedirs(output_dir, exist_ok=True)

    # Default model_path to weights/best.pth in task dir
    if args.model_path is None:
        args.model_path = os.path.join(output_dir, 'weights', 'best.pth')

    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {args.gpu_id}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    patch_size = checkpoint.get('img_height', 128)
    in_channels = checkpoint.get('in_channels', 1)
    print(f"Model patch size: {patch_size}, in_channels: {in_channels}")

    model_module = importlib.import_module(args.model_file)
    SegmentationNetwork = model_module.SegmentationNetwork
    base_channels = args.base_channels
    model = SegmentationNetwork(in_channels=in_channels, out_channels=2, base_channels=base_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    dataset = InferenceDataset(images_dir=args.images_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_scores = []
    all_labels = []
    pixel_aurocs = []
    all_spots = []  # (filename, x, y, score) across all images

    for i, sample in enumerate(dataloader):
        image = sample['image'].squeeze().numpy()
        img_path = sample['image_path'][0]
        h, w = image.shape[:2]

        heatmap, processed_image = sliding_window_inference(
            image, model, patch_size, in_channels, device
        )
        print(f"\r  Processing [{i+1}/{len(dataloader)}] {os.path.basename(img_path)}", end='', flush=True)

        # Load GT mask if labels_dir provided
        gt_mask = None
        if args.labels_dir:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(args.labels_dir, basename + '.txt')
            gt_mask = yolo_label_to_mask(label_path, h, w)

        if gt_mask is not None:
            max_score = heatmap.max()
            gt_binary = gt_mask.astype(np.float32) / 255.0

            has_anomaly = 1.0 if np.sum(gt_binary) > 0 else 0.0
            all_scores.append(max_score)
            all_labels.append(has_anomaly)

            # Per-image pixel AUROC (avoids OOM from accumulating all pixels)
            if len(np.unique(gt_binary)) > 1:
                pa = roc_auc_score(gt_binary.flatten(), heatmap.flatten())
                pixel_aurocs.append(pa)

        # Extract spots for review efficiency
        if args.csv_path:
            img_filename = os.path.basename(img_path)
            spots = extract_spots(heatmap, threshold=args.threshold)
            for sx, sy, sc in spots:
                all_spots.append((img_filename, sx, sy, sc))

    print()

    if args.labels_dir and len(all_scores) > 0:
        if len(np.unique(all_labels)) > 1:
            image_auroc = roc_auc_score(all_labels, all_scores)
            print(f"\nImage-level AUROC: {image_auroc:.4f}")
        else:
            print("\nImage-level AUROC: Cannot calculate (only one class present)")

        if len(pixel_aurocs) > 0:
            mean_pixel_auroc = np.mean(pixel_aurocs)
            print(f"Pixel-level AUROC (mean per-image): {mean_pixel_auroc:.4f} "
                  f"({len(pixel_aurocs)} images with defects)")
        else:
            print("Pixel-level AUROC: Cannot calculate (no images with both classes)")

    # Review efficiency analysis
    if args.csv_path:
        gt_defects = load_defect_csv(args.csv_path)
        all_spots.sort(key=lambda s: s[3], reverse=True)
        print("Review efficiency:")
        plot_review_efficiency(all_spots, gt_defects,
                               os.path.join(output_dir, 'review_efficiency_all.png'),
                               title='Review Efficiency (All)')
        plot_review_efficiency(all_spots, gt_defects,
                               os.path.join(output_dir, 'review_efficiency_dsnr_low.png'),
                               title='Review Efficiency (DSNR < 3.5)',
                               dsnr_filter='low')
        plot_review_efficiency(all_spots, gt_defects,
                               os.path.join(output_dir, 'review_efficiency_dsnr_high.png'),
                               title='Review Efficiency (DSNR >= 3.5)',
                               dsnr_filter='high')

    print(f"\nInference completed. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Inference for SODUnet')

    parser.add_argument('--task_name', type=str, required=True,
                        help='Task name (outputs saved to ../outputs/<task_name>/)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to checkpoint (default: ../outputs/<task_name>/best.pth)')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to test images directory')
    parser.add_argument('--labels_dir', type=str, default=None,
                        help='Path to YOLO labels directory for GT (optional)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use. Set to -1 for CPU (default: 0)')
    parser.add_argument('--model_file', type=str, default='model_v1',
                        help='Model module: model_v1, model_v2, model_v3, model_v4 (default: model_v1)')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channel width (default: 64)')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Path to test_defects.csv for review efficiency analysis')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Heatmap threshold for spot extraction (default: 0.1)')

    args = parser.parse_args()

    inference(args)


if __name__ == "__main__":
    main()
