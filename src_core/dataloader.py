import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from glob import glob


def yolo_label_to_mask(label_path, img_h, img_w):
    """Convert YOLO polygon txt to binary mask on-the-fly."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + at least 3 points (6 coords)
                continue
            coords = list(map(float, parts[1:]))
            pts = np.array(coords).reshape(-1, 2)
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h
            pts = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


def yolo_label_to_centroids(label_path, img_h, img_w):
    """Return list of (cx, cy) pixel coordinates from YOLO polygon label."""
    centroids = []
    if not os.path.exists(label_path):
        return centroids
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = list(map(float, parts[1:]))
            pts = np.array(coords).reshape(-1, 2)
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h
            cx, cy = np.mean(pts, axis=0)
            centroids.append((cx, cy))
    return centroids


def calculate_positions(img_size, patch_size, min_patches=2):
    """Calculate patch positions with minimum overlap, maximum coverage."""
    max_start = img_size - patch_size
    if max_start < 0:
        return None
    elif max_start == 0:
        return [0]
    else:
        num_patches = max(min_patches, int(np.ceil(img_size / patch_size)))
        positions = np.linspace(0, max_start, num_patches).astype(int)
        return positions.tolist()


class DefectDataset(Dataset):
    """
    Dataset for defect segmentation training.
    Reads images + YOLO polygon labels, rasterizes masks on-the-fly.
    Cuts images into fixed-size patches via sliding window.

    bg_ratio controls background patch sampling:
      -1 (default): use all patches (original behavior)
       0: defect patches only
       N: N background patches per defect patch
    """

    def __init__(self, images_dir, labels_dir, patch_size=128, in_channels=1,
                 bg_ratio=-1):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.bg_ratio = bg_ratio

        # Collect image paths
        self.image_paths = sorted(
            glob(os.path.join(images_dir, "*.png"))
            + glob(os.path.join(images_dir, "*.jpg"))
            + glob(os.path.join(images_dir, "*.PNG"))
            + glob(os.path.join(images_dir, "*.JPG"))
            + glob(os.path.join(images_dir, "*.tiff"))
            + glob(os.path.join(images_dir, "*.tif"))
            + glob(os.path.join(images_dir, "*.TIFF"))
            + glob(os.path.join(images_dir, "*.TIF"))
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {images_dir}")

        # Auto-detect image size from first image
        sample = cv2.imread(self.image_paths[0], cv2.IMREAD_GRAYSCALE)
        self.img_h, self.img_w = sample.shape[:2]

        # Calculate patch positions
        self.y_positions = calculate_positions(self.img_h, patch_size)
        self.x_positions = calculate_positions(self.img_w, patch_size)
        if self.y_positions is None or self.x_positions is None:
            raise ValueError(
                f"Image {self.img_h}x{self.img_w} smaller than patch {patch_size}"
            )

        self.patches_per_image = len(self.y_positions) * len(self.x_positions)
        self.total_patches = len(self.image_paths) * self.patches_per_image

        # Biased sampling: classify patches as defect or background
        self.samples = None  # None means use all patches
        if bg_ratio >= 0:
            self.defect_patches = []  # list of (img_idx, patch_idx)
            self.bg_patches = []
            for img_idx, img_path in enumerate(self.image_paths):
                label_path = self._get_label_path(img_path)
                centroids = yolo_label_to_centroids(label_path, self.img_h, self.img_w)
                for patch_idx in range(self.patches_per_image):
                    y_idx = patch_idx // len(self.x_positions)
                    x_idx = patch_idx % len(self.x_positions)
                    sy = self.y_positions[y_idx]
                    sx = self.x_positions[x_idx]
                    ey = sy + patch_size
                    ex = sx + patch_size
                    has_defect = any(sy <= cy < ey and sx <= cx < ex
                                    for cx, cy in centroids)
                    if has_defect:
                        self.defect_patches.append((img_idx, patch_idx))
                    else:
                        self.bg_patches.append((img_idx, patch_idx))
            self._build_sample_list()
            print(f"Biased sampling: bg_ratio={bg_ratio}, "
                  f"{len(self.defect_patches)} defect patches, "
                  f"{len(self.bg_patches)} bg patches, "
                  f"{len(self.samples)} samples/epoch")
        else:
            print(f"Images: {len(self.image_paths)}, size: {self.img_h}x{self.img_w}")
            print(f"Patch: {patch_size}x{patch_size}, "
                  f"Y:{len(self.y_positions)} X:{len(self.x_positions)}, "
                  f"total: {self.total_patches}")

    def _build_sample_list(self):
        """Build sample list with biased defect/bg ratio."""
        samples = list(self.defect_patches)
        if self.bg_ratio > 0 and len(self.bg_patches) > 0:
            n_bg = int(len(self.defect_patches) * self.bg_ratio)
            n_bg = min(n_bg, len(self.bg_patches))
            bg_idx = np.random.choice(len(self.bg_patches), n_bg, replace=False)
            samples.extend(self.bg_patches[i] for i in bg_idx)
        self.samples = samples

    def resample(self):
        """Re-randomize background patches for a new epoch. No-op if bg_ratio < 0."""
        if self.bg_ratio >= 0:
            self._build_sample_list()

    def __len__(self):
        if self.samples is not None:
            return len(self.samples)
        return self.total_patches

    def _get_label_path(self, img_path):
        """Get corresponding YOLO label txt path."""
        basename = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.labels_dir, basename + ".txt")

    def __getitem__(self, idx):
        if self.samples is not None:
            img_idx, patch_idx = self.samples[idx]
            y_idx = patch_idx // len(self.x_positions)
            x_idx = patch_idx % len(self.x_positions)
        else:
            img_idx = idx // self.patches_per_image
            patch_idx = idx % self.patches_per_image
            y_idx = patch_idx // len(self.x_positions)
            x_idx = patch_idx % len(self.x_positions)

        img_path = self.image_paths[img_idx]
        sy = self.y_positions[y_idx]
        sx = self.x_positions[x_idx]
        ey = sy + self.patch_size
        ex = sx + self.patch_size

        # Load grayscale image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        # Load mask from YOLO label
        label_path = self._get_label_path(img_path)
        mask = yolo_label_to_mask(label_path, self.img_h, self.img_w)

        # Crop patch
        image_patch = image[sy:ey, sx:ex]
        mask_patch = mask[sy:ey, sx:ex]

        # To tensor: image [C, H, W] normalized to 0-1
        if self.in_channels == 1:
            img_tensor = torch.from_numpy(image_patch).float().unsqueeze(0) / 255.0
        else:
            img_tensor = torch.from_numpy(
                np.stack([image_patch] * self.in_channels, axis=0)
            ).float() / 255.0

        # Mask: binary [1, H, W]
        mask_tensor = torch.from_numpy(
            (mask_patch > 0).astype(np.float32)
        ).unsqueeze(0)

        return {"image": img_tensor, "mask": mask_tensor}
