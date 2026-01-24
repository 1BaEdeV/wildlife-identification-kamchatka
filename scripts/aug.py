from pathlib import Path
import os
import cv2
import numpy as np
import albumentations as A

INPUT_DIR = Path(r'E:\ULTIMATE_PROJECT\wildlife-identification-kamchatka\full_dataset\Nycticebus_menagensis\all_photos')
OUTPUT_DIR = Path(r'E:\ULTIMATE_PROJECT\wildlife-identification-kamchatka\full_dataset\Nycticebus_menagensis\all_photos\augmented')
NUM_AUGMENTATIONS_PER_IMAGE = 4

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    return img

def imwrite_unicode(path: Path, img, params=None):
    params = params or []
    ext = path.suffix if path.suffix else ".jpg"
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    buf.tofile(str(path))
    return True

def build_augmenter():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Affine(
    translate_percent=(-0.06, 0.06),
    scale=(0.85, 1.15),
    rotate=(-20, 20),
    border_mode=cv2.BORDER_REFLECT_101,
    p=0.6
),
        A.GaussianBlur(blur_limit=(3, 7), p=0.25),
        A.ImageCompression(quality_range=(50, 95), p=0.25),
        A.HueSaturationValue(p=0.35),
        A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(0.05, 0.2),
                        hole_width_range=(0.05, 0.2), p=0.25),
    ])

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aug = build_augmenter()

    images = [p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    for img_path in images:
        img_bgr = imread_unicode(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        stem = img_path.stem
        ext = img_path.suffix.lower()

        for k in range(NUM_AUGMENTATIONS_PER_IMAGE):
            aug_img = aug(image=img_bgr)["image"]  # Albumentations возвращает dict [web:4]
            out_path = OUTPUT_DIR / f"{stem}__aug{k+1}{ext}"
            imwrite_unicode(out_path, aug_img)

if __name__ == "__main__":
    main()
