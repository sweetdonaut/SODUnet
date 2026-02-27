"""
Convert DAGM 2007 dataset to YOLO segmentation format.

Output structure:
  DAGM_yolo/
  ├── dataset.yaml
  ├── images/
  │   ├── train/    # All training images (renamed: c{class}_{id}.png)
  │   └── test/
  ├── labels/
  │   ├── train/    # YOLO polygon txt files
  │   └── test/
  └── masks/
      ├── train/    # Original binary masks (for UNet)
      └── test/
"""

import cv2
import numpy as np
import os
import shutil
import yaml


SRC = "DAGM_KaggleUpload"
DST = "DAGM_yolo"
IMG_SIZE = 512  # DAGM images are 512x512
MIN_CONTOUR_AREA = 5  # Ignore tiny noise contours


def mask_to_yolo_polygons(mask_path):
    """Convert a binary mask PNG to YOLO segmentation polygon lines."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        # Flatten contour points and normalize to 0-1
        pts = cnt.reshape(-1, 2).astype(np.float64)
        pts[:, 0] /= IMG_SIZE  # x
        pts[:, 1] /= IMG_SIZE  # y
        # YOLO format: class_id x1 y1 x2 y2 ...
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        lines.append(f"0 {coords}")

    return lines


def convert():
    # Create output directories
    for split in ["train", "test"]:
        os.makedirs(os.path.join(DST, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DST, "labels", split), exist_ok=True)
        os.makedirs(os.path.join(DST, "masks", split), exist_ok=True)

    stats = {"train": {"images": 0, "defect": 0, "normal": 0, "polygons": 0},
             "test":  {"images": 0, "defect": 0, "normal": 0, "polygons": 0}}

    for cls_name in sorted(os.listdir(SRC)):
        cls_path = os.path.join(SRC, cls_name)
        if not os.path.isdir(cls_path):
            continue

        cls_num = cls_name.replace("Class", "")  # "1", "2", ...

        for dagm_split, yolo_split in [("Train", "train"), ("Test", "test")]:
            split_path = os.path.join(cls_path, dagm_split)
            label_dir = os.path.join(split_path, "Label")

            # Build set of images that have labels
            label_files = {}
            if os.path.exists(label_dir):
                for lf in os.listdir(label_dir):
                    if lf.endswith(".PNG") or lf.endswith(".png"):
                        # e.g. "0595_label.PNG" -> "0595"
                        img_id = lf.split("_label")[0]
                        label_files[img_id] = os.path.join(label_dir, lf)

            # Process all images
            for f in sorted(os.listdir(split_path)):
                if not (f.endswith(".PNG") or f.endswith(".png")):
                    continue
                if f == "Thumbs.db":
                    continue

                img_id = f.split(".")[0]  # e.g. "0595"
                new_name = f"c{cls_num}_{img_id}"

                # Copy image
                src_img = os.path.join(split_path, f)
                dst_img = os.path.join(DST, "images", yolo_split, f"{new_name}.png")
                shutil.copy2(src_img, dst_img)
                stats[yolo_split]["images"] += 1

                # Process label
                txt_path = os.path.join(DST, "labels", yolo_split, f"{new_name}.txt")
                mask_dst = os.path.join(DST, "masks", yolo_split, f"{new_name}.png")

                if img_id in label_files:
                    # Has defect - convert mask to polygon
                    mask_src = label_files[img_id]
                    shutil.copy2(mask_src, mask_dst)

                    lines = mask_to_yolo_polygons(mask_src)
                    with open(txt_path, "w") as fp:
                        fp.write("\n".join(lines) + "\n" if lines else "")

                    stats[yolo_split]["defect"] += 1
                    stats[yolo_split]["polygons"] += len(lines)
                else:
                    # Normal image - empty label file, black mask
                    with open(txt_path, "w") as fp:
                        pass  # empty file
                    # Create black mask
                    black = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                    cv2.imwrite(mask_dst, black)
                    stats[yolo_split]["normal"] += 1

    # Write dataset.yaml
    yaml_content = {
        "path": os.path.abspath(DST),
        "train": "images/train",
        "val": "images/test",
        "nc": 1,
        "names": ["defect"],
    }
    yaml_path = os.path.join(DST, "dataset.yaml")
    with open(yaml_path, "w") as fp:
        yaml.dump(yaml_content, fp, default_flow_style=False)

    # Print summary
    print("=" * 60)
    print("DAGM 2007 -> YOLO Segmentation Conversion Complete")
    print("=" * 60)
    for split in ["train", "test"]:
        s = stats[split]
        print(f"\n{split}:")
        print(f"  Images:   {s['images']}")
        print(f"  Defect:   {s['defect']} ({s['polygons']} polygons)")
        print(f"  Normal:   {s['normal']}")
    print(f"\nOutput: {os.path.abspath(DST)}")
    print(f"YAML:   {os.path.abspath(yaml_path)}")


if __name__ == "__main__":
    convert()
