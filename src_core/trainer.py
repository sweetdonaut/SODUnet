import torch
from torch.utils.data import DataLoader
from torch import optim
import os
import argparse
import csv
import numpy as np
from datetime import datetime
import cv2
import glob
import random
from loss import FocalLoss
import importlib
import inspect
from dataloader import DefectDataset, calculate_positions, yolo_label_to_centroids
from inference import extract_spots

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_focal_gamma(epoch, total_epochs, gamma_start, gamma_end, schedule='cosine'):
    """
    Calculate gamma value for current epoch using cosine schedule
    Args:
        epoch: current epoch (0-indexed)
        total_epochs: total number of epochs
        gamma_start: initial gamma value
        gamma_end: final gamma value
        schedule: 'linear' or 'cosine'
    Returns:
        current gamma value
    """
    if schedule == 'linear':
        gamma = gamma_start + (gamma_end - gamma_start) * (epoch / total_epochs)
    elif schedule == 'cosine':
        import math
        progress = epoch / total_epochs
        gamma = gamma_start + (gamma_end - gamma_start) * (1 - math.cos(progress * math.pi)) / 2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return gamma

def evaluate_model(model, images_dir, labels_dir, in_channels, patch_size, device,
                   match_tol=15, spot_threshold=0.05):
    """Evaluate model on validation set using FROC metrics.

    Returns (recall_at_1fp, mean_loc_error).
    """
    model.eval()

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png"))
        + glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.PNG"))
        + glob.glob(os.path.join(images_dir, "*.JPG"))
        + glob.glob(os.path.join(images_dir, "*.tiff"))
        + glob.glob(os.path.join(images_dir, "*.tif"))
        + glob.glob(os.path.join(images_dir, "*.TIFF"))
        + glob.glob(os.path.join(images_dir, "*.TIF"))
    )

    all_spots = []   # (x, y, score) across all images
    all_gt = []       # (x, y) across all images
    num_images = 0

    with torch.no_grad():
        for img_path in image_paths:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            h, w = image.shape[:2]

            if h < patch_size or w < patch_size:
                continue

            full_score_map = np.zeros((h, w), dtype=np.float32)
            count_map = np.zeros((h, w), dtype=np.float32)

            y_positions = calculate_positions(h, patch_size)
            x_positions = calculate_positions(w, patch_size)

            if y_positions is None or x_positions is None:
                continue

            for y in y_positions:
                for x in x_positions:
                    patch = image[y:y+patch_size, x:x+patch_size].astype(np.float32)

                    if in_channels == 1:
                        input_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float() / 255.0
                    else:
                        patch_3ch = np.stack([patch] * in_channels, axis=0)
                        input_tensor = torch.from_numpy(patch_3ch).unsqueeze(0).float() / 255.0
                    input_tensor = input_tensor.to(device)

                    output = model(input_tensor)
                    output_sm = torch.softmax(output, dim=1)
                    patch_score = output_sm[:, 1, :, :].squeeze().cpu().numpy()

                    full_score_map[y:y+patch_size, x:x+patch_size] += patch_score
                    count_map[y:y+patch_size, x:x+patch_size] += 1

            heatmap = full_score_map / np.maximum(count_map, 1)
            num_images += 1

            # Extract predicted spots
            spots = extract_spots(heatmap, threshold=spot_threshold)
            all_spots.extend(spots)

            # Load GT centroids from YOLO label
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, basename + ".txt")
            gt_centroids = yolo_label_to_centroids(label_path, h, w)
            all_gt.extend(gt_centroids)

    # Compute FROC metrics
    total_gt = len(all_gt)
    recall_at_1fp = 0.0
    mean_loc_error = float('nan')

    if total_gt > 0 and len(all_spots) > 0:
        # Sort spots by score descending
        all_spots.sort(key=lambda s: s[2], reverse=True)

        gt_matched = [False] * total_gt
        matched_distances = []
        sensitivities = []
        fp_per_image = []
        matched_count = 0
        fp_count = 0
        reached_1fp = False

        for sx, sy, score in all_spots:
            best_j = -1
            best_dist = float('inf')
            for j, (gx, gy) in enumerate(all_gt):
                if not gt_matched[j] and abs(sx - gx) < match_tol and abs(sy - gy) < match_tol:
                    dist = np.sqrt((sx - gx)**2 + (sy - gy)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j

            if best_j >= 0:
                gt_matched[best_j] = True
                matched_count += 1
                if not reached_1fp:
                    matched_distances.append(best_dist)
            else:
                fp_count += 1

            sensitivities.append(matched_count / total_gt)
            fp_per_image.append(fp_count / num_images)

            if not reached_1fp and fp_count / num_images >= 1.0:
                reached_1fp = True

        # Interpolate recall at 1 FP/image
        for k in range(len(fp_per_image)):
            if fp_per_image[k] >= 1.0:
                if k == 0:
                    recall_at_1fp = sensitivities[k]
                else:
                    # Linear interpolation
                    fp_prev, fp_curr = fp_per_image[k-1], fp_per_image[k]
                    s_prev, s_curr = sensitivities[k-1], sensitivities[k]
                    t = (1.0 - fp_prev) / (fp_curr - fp_prev)
                    recall_at_1fp = s_prev + t * (s_curr - s_prev)
                break
        else:
            # Never reached 1 FP/image — all spots were matched
            recall_at_1fp = sensitivities[-1] if sensitivities else 0.0

        if matched_distances:
            mean_loc_error = np.mean(matched_distances)

    model.train()

    return recall_at_1fp, mean_loc_error

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_on_device(args):
    output_dir = os.path.join('..', 'outputs', args.task_name)
    weights_dir = os.path.join(output_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # Initialize results CSV
    csv_path = os.path.join(output_dir, 'results.csv')
    csv_fields = ['epoch', 'avg_loss', 'lr', 'gamma', 'recall_at_1fp', 'mean_loc_error']
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Set random seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {args.seed}")

    # Set up device
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {args.gpu_id}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    patch_size = args.patch_size

    run_name = f'SODUnet_lr{args.lr}_ep{args.epochs}_bs{args.bs}_{patch_size}x{patch_size}'

    # Segmentation network
    model_module = importlib.import_module(args.model_file)
    SegmentationNetwork = model_module.SegmentationNetwork
    model_kwargs = dict(
        in_channels=args.in_channels,
        out_channels=2,
        base_channels=getattr(args, 'base_channels', 64),
        dropout=getattr(args, 'dropout', 0.0),
        use_resblock=getattr(args, 'use_resblock', False),
        encoder_attention=getattr(args, 'encoder_attention', False),
        use_fpn=getattr(args, 'use_fpn', False),
    )
    sig = inspect.signature(SegmentationNetwork.__init__)
    valid_kwargs = {k: v for k, v in model_kwargs.items() if k in sig.parameters}
    model_seg = SegmentationNetwork(**valid_kwargs)
    total_params = sum(p.numel() for p in model_seg.parameters())
    print(f"Model params: {total_params:,} ({total_params/1e6:.2f}M)")
    model_seg.to(device)
    model_seg.apply(weights_init)

    optimizer = torch.optim.Adam(model_seg.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        [int(args.epochs*0.8), int(args.epochs*0.9)],
        gamma=0.2,
        last_epoch=-1
    )

    # Initialize loss function with alpha=0.75 for defect class balance
    criterion = FocalLoss(alpha=0.75, gamma=args.gamma_start)
    print(f"Using Focal Loss with alpha=0.75, gamma schedule: [{args.gamma_start}, {args.gamma_end}] (cosine)")

    dataset = DefectDataset(
        images_dir=os.path.join(args.data_root, 'train', 'images'),
        labels_dir=os.path.join(args.data_root, 'train', 'labels'),
        patch_size=patch_size,
        in_channels=args.in_channels,
    )

    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=7)
    print(f"Dataset size: {len(dataset)} samples per epoch")

    # Count validation data
    valid_images_dir = os.path.join(args.data_root, 'valid', 'images')
    valid_labels_dir = os.path.join(args.data_root, 'valid', 'labels')
    num_valid_images = 0
    num_valid_patches = 0
    if os.path.exists(valid_images_dir):
        valid_paths = sorted(
            glob.glob(os.path.join(valid_images_dir, "*.png"))
            + glob.glob(os.path.join(valid_images_dir, "*.jpg"))
            + glob.glob(os.path.join(valid_images_dir, "*.PNG"))
            + glob.glob(os.path.join(valid_images_dir, "*.JPG"))
            + glob.glob(os.path.join(valid_images_dir, "*.tiff"))
            + glob.glob(os.path.join(valid_images_dir, "*.tif"))
            + glob.glob(os.path.join(valid_images_dir, "*.TIFF"))
            + glob.glob(os.path.join(valid_images_dir, "*.TIF"))
        )
        num_valid_images = len(valid_paths)
        num_valid_patches = num_valid_images * dataset.patches_per_image

    # Write training log
    log_path = os.path.join(output_dir, 'training_log.txt')
    with open(log_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"  SODUnet Training Log\n")
        f.write(f"{'='*60}\n")
        f.write(f"Date:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Task:             {args.task_name}\n\n")
        f.write(f"--- Model ---\n")
        f.write(f"Model file:       {args.model_file}\n")
        f.write(f"Base channels:    {getattr(args, 'base_channels', 64)}\n")
        f.write(f"Parameters:       {total_params:,} ({total_params/1e6:.2f}M)\n")
        f.write(f"Dropout:          {getattr(args, 'dropout', 0.0)}\n")
        f.write(f"ResBlock:         {getattr(args, 'use_resblock', False)}\n")
        f.write(f"Encoder Attn:     {getattr(args, 'encoder_attention', False)}\n")
        f.write(f"FPN:              {getattr(args, 'use_fpn', False)}\n\n")
        f.write(f"--- Data ---\n")
        f.write(f"Data root:        {args.data_root}\n")
        f.write(f"Image size:       {dataset.img_h}x{dataset.img_w}\n")
        f.write(f"Train images:     {len(dataset.image_paths)}\n")
        f.write(f"Valid images:     {num_valid_images}\n")
        f.write(f"Patch size:       {patch_size}x{patch_size}\n")
        f.write(f"Patches/image:    {dataset.patches_per_image}\n")
        f.write(f"Train patches:    {dataset.total_patches}\n")
        f.write(f"Valid patches:    {num_valid_patches}\n")
        f.write(f"In channels:      {args.in_channels}\n\n")
        f.write(f"--- Training ---\n")
        f.write(f"Batch size:       {args.bs}\n")
        f.write(f"Learning rate:    {args.lr}\n")
        f.write(f"Epochs:           {args.epochs}\n")
        f.write(f"Eval interval:    {args.eval_interval}\n")
        f.write(f"Gamma schedule:   [{args.gamma_start}, {args.gamma_end}] (cosine)\n")
        f.write(f"Seed:             {args.seed}\n")
        f.write(f"Device:           {device}\n\n")
        f.write(f"--- Evaluation ---\n")
        f.write(f"Match tolerance:  15 px (Chebyshev)\n")
        f.write(f"Spot threshold:   0.05\n")
        f.write(f"{'='*60}\n")
    print(f"Training log saved: {log_path}")

    num_batches = len(dataloader)

    best_recall = -1.0
    best_epoch = -1

    for epoch in range(args.epochs):
        # Update focal loss gamma for current epoch
        current_gamma = get_focal_gamma(epoch, args.epochs, args.gamma_start, args.gamma_end, schedule='cosine')
        criterion.update_params(gamma=current_gamma)

        epoch_loss = 0.0

        for i_batch, sample_batched in enumerate(dataloader):
            input_image = sample_batched["image"].to(device)
            target_mask = sample_batched["mask"].to(device)

            # Forward pass through segmentation network
            output = model_seg(input_image)

            # Handle deep supervision (model returns tuple) or single output
            if isinstance(output, tuple):
                out_mask, aux_d3, aux_d2 = output
                out_mask_sm = torch.softmax(out_mask, dim=1)
                loss_main = criterion(out_mask_sm, target_mask)
                loss_aux_d3 = criterion(torch.softmax(aux_d3, dim=1), target_mask)
                loss_aux_d2 = criterion(torch.softmax(aux_d2, dim=1), target_mask)
                loss = loss_main + 0.4 * loss_aux_d3 + 0.4 * loss_aux_d2
            else:
                out_mask_sm = torch.softmax(output, dim=1)
                loss = criterion(out_mask_sm, target_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Print training progress
            if i_batch % 10 == 0 or i_batch == num_batches - 1:
                current_lr = get_lr(optimizer)
                progress = (i_batch + 1) / num_batches * 100
                print(f'\rEpoch [{epoch+1}/{args.epochs}] - Batch [{i_batch+1}/{num_batches}] ({progress:.1f}%) - '
                      f'Loss: {loss.item():.4e} - LR: {current_lr:.6f}', end='', flush=True)

        scheduler.step()

        # Print epoch summary
        avg_loss = epoch_loss / num_batches
        print(f'\nEpoch [{epoch+1}/{args.epochs}] Summary - Avg Loss: {avg_loss:.4e} - Gamma: {current_gamma:.3f}', end='')

        # Evaluate on validation set every eval_interval epochs (and always on last epoch)
        recall_at_1fp, mean_loc_error = None, None
        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epochs:
            valid_images_dir = os.path.join(args.data_root, 'valid', 'images')
            valid_labels_dir = os.path.join(args.data_root, 'valid', 'labels')

            if os.path.exists(valid_images_dir):
                recall_at_1fp, mean_loc_error = evaluate_model(
                    model_seg, valid_images_dir, valid_labels_dir,
                    args.in_channels, patch_size, device
                )
                loc_str = f'{mean_loc_error:.1f}px' if not np.isnan(mean_loc_error) else 'N/A'
                print(f' - Recall@1FP: {recall_at_1fp:.4f} - Loc Error: {loc_str}', end='')
            else:
                print(' - Validation data not found', end='')
        print()

        # Log to CSV
        csv_writer.writerow({
            'epoch': epoch + 1,
            'avg_loss': f'{avg_loss:.6e}',
            'lr': f'{get_lr(optimizer):.6f}',
            'gamma': f'{current_gamma:.3f}',
            'recall_at_1fp': f'{recall_at_1fp:.4f}' if recall_at_1fp is not None else '',
            'mean_loc_error': f'{mean_loc_error:.2f}' if mean_loc_error is not None and not np.isnan(mean_loc_error) else '',
        })
        csv_file.flush()

        # Build checkpoint dict
        checkpoint = {
            'model_state_dict': model_seg.state_dict(),
            'img_height': patch_size,
            'img_width': patch_size,
            'in_channels': args.in_channels,
            'epoch': epoch,
            'seed': args.seed,
            'recall_at_1fp': recall_at_1fp,
            'mean_loc_error': mean_loc_error,
        }

        # Always save last model
        torch.save(checkpoint, os.path.join(weights_dir, 'last.pth'))

        # Save best model (by recall@1FP)
        if recall_at_1fp is not None and recall_at_1fp > best_recall:
            best_recall = recall_at_1fp
            best_epoch = epoch + 1
            torch.save(checkpoint, os.path.join(weights_dir, 'best.pth'))
            print(f'  >> New best model saved (Recall@1FP: {best_recall:.4f} @ epoch {best_epoch})')

    csv_file.close()
    print(f'\nTraining complete. Best Recall@1FP: {best_recall:.4f} @ epoch {best_epoch}')
    print(f'Results saved to: {csv_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', action='store', type=int, required=True, help='Batch size')
    parser.add_argument('--lr', action='store', type=float, required=True, help='Learning rate')
    parser.add_argument('--epochs', action='store', type=int, required=True, help='Number of epochs')
    parser.add_argument('--gpu_id', action='store', type=int, default=0,
                        help='GPU ID to use. Set to -1 to use CPU')
    parser.add_argument('--task_name', type=str, required=True,
                        help='Task name (outputs saved to ../outputs/<task_name>/)')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size for training (default: 128)')
    parser.add_argument('--data_root', action='store', type=str, required=True,
                        help='Dataset root directory (contains train/ and valid/)')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--gamma_start', type=float, default=1.0,
                        help='Starting gamma value for focal loss (default: 1.0)')
    parser.add_argument('--gamma_end', type=float, default=3.0,
                        help='Ending gamma value for focal loss (default: 3.0)')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Evaluate every N epochs (default: 1)')
    parser.add_argument('--model_file', type=str, default='model_v1',
                        help='Model module: model_v1, model_v2, model_v3, model_v4 (default: model_v1)')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channel width for UNet encoder (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout2d rate in decoder (default: 0.0)')
    parser.add_argument('--use_resblock', action='store_true',
                        help='Use residual blocks in encoder')
    parser.add_argument('--encoder_attention', action='store_true',
                        help='Add CoordAttention on encoder outputs')
    parser.add_argument('--use_fpn', action='store_true',
                        help='Add FPN bridge between encoder and decoder')

    args = parser.parse_args()

    train_on_device(args)

if __name__ == "__main__":
    main()
