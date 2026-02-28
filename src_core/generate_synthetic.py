"""Generate synthetic 1536x1536 grayscale images with Gaussian spot defects (<10x10 px)."""
import os
import numpy as np
import cv2
import random
import argparse
import csv
import math


NOISE_STD = 15  # background noise standard deviation


def make_gaussian_spot(sigma, amplitude, patch_radius):
    """Create a 2D Gaussian spot (intensity drop values)."""
    coords = np.arange(-patch_radius, patch_radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing='ij')
    g = amplitude * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return g


def generate_dataset(output_dir, num_train=100, num_valid=20, num_test=10,
                     img_size=1536, fmt='png', seed=42):
    random.seed(seed)
    np.random.seed(seed)

    splits = [('train', num_train), ('valid', num_valid), ('test', num_test)]

    for split_name, num_images in splits:
        img_dir = os.path.join(output_dir, split_name, 'images')
        lbl_dir = os.path.join(output_dir, split_name, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        csv_rows = []

        for i in range(num_images):
            # Background: random base intensity + Gaussian noise
            base = np.random.randint(100, 180)
            img = np.random.normal(base, NOISE_STD, (img_size, img_size)).astype(np.float32)

            # Add slight gradient for realism
            grad = np.linspace(-10, 10, img_size).reshape(1, -1)
            img += grad

            # Gaussian spot parameters
            # sigma 0.5-1.5 → effective diameter (6*sigma) = 3-9 pixels, all < 10
            sigma = np.random.uniform(0.5, 1.5)
            patch_radius = min(int(np.ceil(3 * sigma)), 4)  # max 4 → patch 9x9 < 10x10

            # Amplitude: DSNR range ~1.5 to 8 (amplitude = DSNR * NOISE_STD)
            dsnr = np.random.uniform(1.5, 8.0)
            amplitude = dsnr * NOISE_STD

            # Random position
            margin = patch_radius + 5
            cx = random.randint(margin, img_size - margin - 1)
            cy = random.randint(margin, img_size - margin - 1)

            # Apply Gaussian spot (darker than background)
            spot = make_gaussian_spot(sigma, amplitude, patch_radius)
            y1 = cy - patch_radius
            y2 = cy + patch_radius + 1
            x1 = cx - patch_radius
            x2 = cx + patch_radius + 1
            img[y1:y2, x1:x2] -= spot

            img = np.clip(img, 0, 255).astype(np.uint8)

            # Save image
            fname = f'{split_name}_{i:04d}.{fmt}'
            cv2.imwrite(os.path.join(img_dir, fname), img)

            # YOLO polygon label: circular boundary at 3*sigma, 12 vertices
            defect_r = 3 * sigma
            n_verts = 12
            coords = []
            for v in range(n_verts):
                angle = 2 * math.pi * v / n_verts
                px = (cx + defect_r * math.cos(angle)) / img_size
                py = (cy + defect_r * math.sin(angle)) / img_size
                coords.extend([px, py])

            lbl_fname = f'{split_name}_{i:04d}.txt'
            coord_str = ' '.join(f'{c:.6f}' for c in coords)
            with open(os.path.join(lbl_dir, lbl_fname), 'w') as f:
                f.write(f'0 {coord_str}\n')

            # Collect CSV data for test split
            if split_name == 'test':
                csv_rows.append({
                    'filename': fname,
                    'rawx': cx,
                    'rawy': cy,
                    'dsnr': round(dsnr, 3),
                })

        # Write test CSV
        if split_name == 'test' and csv_rows:
            csv_path = os.path.join(output_dir, split_name, 'test_defects.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'rawx', 'rawy', 'dsnr'])
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f'  CSV: {csv_path}')

        print(f'{split_name}: {num_images} images ({fmt}) -> {img_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--fmt', type=str, default='png', choices=['png', 'tiff'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output_dir, fmt=args.fmt, seed=args.seed)
