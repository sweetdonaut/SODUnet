import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import argparse
import glob
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model import SegmentationNetwork
from dataloader import calculate_positions, yolo_label_to_mask


class InferenceDataset(Dataset):

    def __init__(self, images_dir):
        self.image_paths = sorted(
            glob.glob(os.path.join(images_dir, "*.png"))
            + glob.glob(os.path.join(images_dir, "*.jpg"))
            + glob.glob(os.path.join(images_dir, "*.PNG"))
            + glob.glob(os.path.join(images_dir, "*.JPG"))
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


def inference(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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

    model = SegmentationNetwork(in_channels=in_channels, out_channels=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    dataset = InferenceDataset(images_dir=args.images_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_scores = []
    all_labels = []
    pixel_scores = []
    pixel_labels = []

    for i, sample in enumerate(dataloader):
        image = sample['image'].squeeze().numpy()
        img_path = sample['image_path'][0]
        h, w = image.shape[:2]

        heatmap, processed_image = sliding_window_inference(
            image, model, patch_size, in_channels, device
        )

        # Load GT mask if labels_dir provided
        gt_mask = None
        if args.labels_dir:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(args.labels_dir, basename + '.txt')
            gt_mask = yolo_label_to_mask(label_path, h, w)

        filename = os.path.basename(img_path).split('.')[0]
        output_path = os.path.join(args.output_dir, f'{filename}_result.png')

        visualize_results(processed_image, heatmap, output_path, gt_mask=gt_mask)
        print(f"Saved result: {output_path}")

        if gt_mask is not None:
            max_score = heatmap.max()
            gt_binary = gt_mask.astype(np.float32) / 255.0

            has_anomaly = 1.0 if np.sum(gt_binary) > 0 else 0.0
            all_scores.append(max_score)
            all_labels.append(has_anomaly)

            pixel_scores.extend(heatmap.flatten())
            pixel_labels.extend(gt_binary.flatten())

    if args.labels_dir and len(all_scores) > 0:
        if len(np.unique(all_labels)) > 1:
            image_auroc = roc_auc_score(all_labels, all_scores)
            print(f"\nImage-level AUROC: {image_auroc:.4f}")
        else:
            print("\nImage-level AUROC: Cannot calculate (only one class present)")

        if len(np.unique(pixel_labels)) > 1:
            pixel_auroc = roc_auc_score(pixel_labels, pixel_scores)
            print(f"Pixel-level AUROC: {pixel_auroc:.4f}")
        else:
            print("Pixel-level AUROC: Cannot calculate (only one class present)")

    print(f"\nInference completed. Results saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Inference for SODUnet')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to test images directory')
    parser.add_argument('--labels_dir', type=str, default=None,
                        help='Path to YOLO labels directory for GT (optional)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output visualizations')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use. Set to -1 for CPU (default: 0)')

    args = parser.parse_args()

    inference(args)


if __name__ == "__main__":
    main()
