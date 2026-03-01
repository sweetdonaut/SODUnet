"""Generate synthetic 1536x1536 grayscale images with Gaussian spot defects (2x2 to 8x8 px).

Changes from v1:
- Defect size: 2x2 to 8x8 pixels (sigma = size / 2.355, using FWHM)
- patch_radius no longer capped at 4
- Actual DSNR measured from generated image and saved to CSV
- CSV generated for ALL splits (not just test)
- Normal (clean) images included in valid/test for image-level AUROC
"""
import os
import numpy as np
import cv2
import random
import argparse
import csv
import math


NOISE_STD = 15  # background noise standard deviation
IMG_SIZE = 1536


def make_gaussian_spot(sigma, amplitude, patch_radius):
    """Create a 2D Gaussian spot (intensity drop values)."""
    coords = np.arange(-patch_radius, patch_radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing='ij')
    g = amplitude * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return g


def measure_actual_dsnr(img_float, img_uint8, cx, cy):
    """Measure actual DSNR from the generated image.

    Computes signal drop from float32 image (before clipping) to avoid
    noise and quantization artifacts. Also checks for 8-bit clipping.

    Returns (actual_dsnr, clipped) where clipped=True if center pixel hit 0 or 255.
    """
    h, w = img_float.shape

    # Background: annular ring from float image (more accurate)
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy)**2 + (xx - cx)**2
    mask = (dist2 >= 20**2) & (dist2 < 50**2)
    bg_pixels = img_float[mask].astype(np.float64)
    bg_mean = bg_pixels.mean()
    bg_std = bg_pixels.std()

    # Signal: center pixel drop from background
    center_val = float(img_float[cy, cx])
    signal = abs(center_val - bg_mean)

    if bg_std < 1e-6:
        return 0.0, False

    actual_dsnr = signal / bg_std

    # Check if clipping occurred
    clipped = (img_uint8[cy, cx] == 0) or (img_uint8[cy, cx] == 255)

    return actual_dsnr, clipped


def generate_image(img_size, noise_std):
    """Generate a background image with noise and slight gradient."""
    base = np.random.randint(130, 210)
    img = np.random.normal(base, noise_std, (img_size, img_size)).astype(np.float32)
    grad = np.linspace(-10, 10, img_size).reshape(1, -1)
    img += grad
    return img, base


def add_defect(img, img_size, noise_std):
    """Add a single Gaussian spot defect to the image.

    Returns (cx, cy, sigma, defect_size, nominal_dsnr) or None if placement failed.
    """
    # Defect size: 2 to 8 pixels (visible diameter, defined as FWHM)
    defect_size = np.random.uniform(2.0, 8.0)
    sigma = defect_size / 2.355  # FWHM = 2.355 * sigma

    patch_radius = int(np.ceil(3 * sigma))

    # Random position with margin
    margin = patch_radius + 10
    if margin >= img_size // 2:
        return None
    cx = random.randint(margin, img_size - margin - 1)
    cy = random.randint(margin, img_size - margin - 1)

    # DSNR range ~1.5 to 8, but cap to avoid clipping at 0
    # Center pixel ≈ base + noise; need (center - amplitude) - 3*σ > 0
    local_base = img[cy, cx]  # includes noise, but good estimate
    max_amplitude = max(local_base - 3 * noise_std, noise_std)  # at least 1σ
    max_dsnr = max_amplitude / noise_std
    if max_dsnr < 1.5:
        max_dsnr = 1.5  # ensure valid range
    dsnr = np.random.uniform(1.5, min(8.0, max_dsnr))
    amplitude = dsnr * noise_std

    # Apply Gaussian spot (darker than background)
    spot = make_gaussian_spot(sigma, amplitude, patch_radius)
    y1 = cy - patch_radius
    y2 = cy + patch_radius + 1
    x1 = cx - patch_radius
    x2 = cx + patch_radius + 1
    img[y1:y2, x1:x2] -= spot

    return cx, cy, sigma, defect_size, dsnr


def make_yolo_label(cx, cy, sigma, img_size, n_verts=12):
    """Create YOLO polygon label string for the defect."""
    defect_r = 3 * sigma
    coords = []
    for v in range(n_verts):
        angle = 2 * math.pi * v / n_verts
        px = (cx + defect_r * math.cos(angle)) / img_size
        py = (cy + defect_r * math.sin(angle)) / img_size
        coords.extend([px, py])
    coord_str = ' '.join(f'{c:.6f}' for c in coords)
    return f'0 {coord_str}'


def generate_dataset(output_dir, num_train=200, num_valid_defect=20,
                     num_valid_normal=5, num_test_defect=200,
                     num_test_normal=5, img_size=IMG_SIZE, fmt='png', seed=42):
    random.seed(seed)
    np.random.seed(seed)

    splits = [
        ('train', num_train, 0),
        ('valid', num_valid_defect, num_valid_normal),
        ('test', num_test_defect, num_test_normal),
    ]

    for split_name, num_defect, num_normal in splits:
        img_dir = os.path.join(output_dir, split_name, 'images')
        lbl_dir = os.path.join(output_dir, split_name, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        csv_rows = []
        total = num_defect + num_normal
        idx = 0

        # --- Defect images ---
        for i in range(num_defect):
            img, base = generate_image(img_size, NOISE_STD)
            result = add_defect(img, img_size, NOISE_STD)
            if result is None:
                continue
            cx, cy, sigma, defect_size, nominal_dsnr = result

            img_clipped = np.clip(img, 0, 255).astype(np.uint8)

            # Measure actual DSNR from float image (before clipping)
            actual_dsnr, clipped = measure_actual_dsnr(img, img_clipped, cx, cy)

            fname = f'{split_name}_{idx:04d}.{fmt}'
            cv2.imwrite(os.path.join(img_dir, fname), img_clipped)

            # YOLO polygon label
            label_str = make_yolo_label(cx, cy, sigma, img_size)
            lbl_fname = f'{split_name}_{idx:04d}.txt'
            with open(os.path.join(lbl_dir, lbl_fname), 'w') as f:
                f.write(label_str + '\n')

            csv_rows.append({
                'filename': fname,
                'rawx': cx,
                'rawy': cy,
                'dsnr': round(nominal_dsnr, 3),
                'defect_size': round(defect_size, 2),
                'sigma': round(sigma, 3),
                'actual_dsnr': round(actual_dsnr, 3),
                'clipped': clipped,
                'base_intensity': base,
            })
            idx += 1

        # --- Normal (clean) images ---
        for i in range(num_normal):
            img, base = generate_image(img_size, NOISE_STD)
            img_clipped = np.clip(img, 0, 255).astype(np.uint8)

            fname = f'{split_name}_{idx:04d}.{fmt}'
            cv2.imwrite(os.path.join(img_dir, fname), img_clipped)

            # Empty label file
            lbl_fname = f'{split_name}_{idx:04d}.txt'
            with open(os.path.join(lbl_dir, lbl_fname), 'w') as f:
                pass  # empty

            idx += 1

        # Write CSV for this split
        csv_name = f'{split_name}_defects.csv'
        csv_path = os.path.join(output_dir, split_name, csv_name)
        if csv_rows:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)

        print(f'{split_name}: {num_defect} defect + {num_normal} normal = {idx} images -> {img_dir}')
        if csv_rows:
            # Print DSNR stats
            nominal = [r['dsnr'] for r in csv_rows]
            actual = [r['actual_dsnr'] for r in csv_rows]
            sizes = [r['defect_size'] for r in csv_rows]
            print(f'  Defect size: {min(sizes):.1f} - {max(sizes):.1f} px')
            print(f'  Nominal DSNR: {min(nominal):.2f} - {max(nominal):.2f} (mean {np.mean(nominal):.2f})')
            print(f'  Actual DSNR:  {min(actual):.2f} - {max(actual):.2f} (mean {np.mean(actual):.2f})')

            # Check correlation
            dsnr_diff = [abs(n - a) for n, a in zip(nominal, actual)]
            print(f'  |Nominal - Actual| DSNR: mean={np.mean(dsnr_diff):.2f}, max={max(dsnr_diff):.2f}')
            num_clipped = sum(1 for r in csv_rows if r['clipped'])
            print(f'  Clipped centers: {num_clipped}/{len(csv_rows)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--fmt', type=str, default='png', choices=['png', 'tiff'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train', type=int, default=200)
    parser.add_argument('--num_test', type=int, default=200)
    args = parser.parse_args()
    generate_dataset(args.output_dir, num_train=args.num_train,
                     num_test_defect=args.num_test, fmt=args.fmt, seed=args.seed)
