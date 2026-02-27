"""Cut 224x224 patches from KolektorSDD2 and produce YOLO-format datasets."""
import cv2
import numpy as np
import os
import glob
import random
import shutil


PATCH_SIZE = 224
SEED = 42


def pad_to_min(img, min_size):
    """Reflect-pad image so both dimensions >= min_size."""
    h, w = img.shape[:2]
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h == 0 and pad_w == 0:
        return img
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)


def mask_to_yolo(mask, class_id=0):
    """Binary mask -> YOLO polygon text lines."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        cnt = cv2.approxPolyDP(cnt, 1.0, True)
        if len(cnt) < 3:
            continue
        parts = [str(class_id)]
        for pt in cnt:
            parts.append(f"{pt[0][0] / w:.6f}")
            parts.append(f"{pt[0][1] / h:.6f}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def extract_patches(images_dir, include_clean=True, one_per_image=False):
    """Extract 224x224 patches from KolektorSDD2 directory.

    Args:
        images_dir: directory containing XXXXX.png and XXXXX_GT.png
        include_clean: whether to include patches from clean images
        one_per_image: if True, only one patch per image (for test set)
    Returns:
        list of (patch_img, patch_mask, name)
    """
    random.seed(SEED)
    np.random.seed(SEED)

    all_imgs = sorted([f for f in glob.glob(os.path.join(images_dir, "*.png"))
                       if "_GT" not in f])
    patches = []

    for img_path in all_imgs:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(images_dir, f"{basename}_GT.png")

        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue

        img = pad_to_min(img, PATCH_SIZE)
        gt = pad_to_min(gt, PATCH_SIZE)
        h, w = img.shape[:2]

        has_defect = np.sum(gt > 0) > 0

        if has_defect:
            _, labels_map = cv2.connectedComponents((gt > 0).astype(np.uint8))
            num_comp = labels_map.max()

            comps = list(range(1, num_comp + 1))
            if one_per_image:
                # pick the largest component
                comps = [max(comps, key=lambda c: np.sum(labels_map == c))]

            for comp_id in comps:
                ys, xs = np.where(labels_map == comp_id)
                cy, cx = int(ys.mean()), int(xs.mean())

                jy = random.randint(-PATCH_SIZE // 4, PATCH_SIZE // 4)
                jx = random.randint(-PATCH_SIZE // 4, PATCH_SIZE // 4)
                cy, cx = cy + jy, cx + jx

                y1 = max(0, min(cy - PATCH_SIZE // 2, h - PATCH_SIZE))
                x1 = max(0, min(cx - PATCH_SIZE // 2, w - PATCH_SIZE))

                patches.append((
                    img[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE],
                    gt[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE],
                    f"{basename}_d{comp_id}"
                ))

        elif include_clean:
            y1 = random.randint(0, h - PATCH_SIZE)
            x1 = random.randint(0, w - PATCH_SIZE)
            patches.append((
                img[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE],
                gt[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE],
                f"{basename}_c0"
            ))

    return patches


def save_patches(patches, out_dir):
    """Save patches as images + YOLO labels."""
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    n_def, n_clean = 0, 0
    for patch_img, patch_mask, name in patches:
        cv2.imwrite(os.path.join(img_dir, f"{name}.png"), patch_img)
        has = np.sum(patch_mask > 0) > 0
        txt = mask_to_yolo(patch_mask) if has else ""
        with open(os.path.join(lbl_dir, f"{name}.txt"), "w") as f:
            f.write(txt)
        if has:
            n_def += 1
        else:
            n_clean += 1
    return n_def, n_clean


def write_yaml(out_dir, name):
    yaml_text = f"""path: {os.path.abspath(out_dir)}
train: train/images
val: valid/images

names:
  0: defect
"""
    with open(os.path.join(out_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_text)


def main():
    raw_dir = os.path.dirname(os.path.abspath(__file__))
    raw_train = os.path.join(raw_dir, "KolektorSDD2_raw", "train")
    raw_test = os.path.join(raw_dir, "KolektorSDD2_raw", "test")
    out_root = os.path.join(raw_dir, "KolektorSDD2_patch")

    # --- Test patches (shared, one per image, all included) ---
    print("Extracting test patches (one per image)...")
    test_patches = extract_patches(raw_test, include_clean=True, one_per_image=True)
    print(f"  Test patches: {len(test_patches)}")

    # --- Dataset A: defect_only ---
    print("\nExtracting train patches (defect only)...")
    train_defect = extract_patches(raw_train, include_clean=False)

    out_a = os.path.join(out_root, "defect_only")
    n_def, n_clean = save_patches(train_defect, os.path.join(out_a, "train"))
    print(f"  defect_only/train: {n_def} defect, {n_clean} clean")

    n_def, n_clean = save_patches(test_patches, os.path.join(out_a, "valid"))
    print(f"  defect_only/valid: {n_def} defect, {n_clean} clean")
    write_yaml(out_a, "defect_only")

    # --- Dataset B: mixed ---
    print("\nExtracting train patches (mixed)...")
    train_mixed = extract_patches(raw_train, include_clean=True)

    out_b = os.path.join(out_root, "mixed")
    n_def, n_clean = save_patches(train_mixed, os.path.join(out_b, "train"))
    print(f"  mixed/train: {n_def} defect, {n_clean} clean")

    # Copy same valid set
    n_def, n_clean = save_patches(test_patches, os.path.join(out_b, "valid"))
    print(f"  mixed/valid: {n_def} defect, {n_clean} clean")
    write_yaml(out_b, "mixed")

    print("\nDone!")


if __name__ == "__main__":
    main()
