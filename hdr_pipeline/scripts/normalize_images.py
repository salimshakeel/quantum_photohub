from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ExifTags


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def list_image_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS])


def apply_exif_orientation(img: Image.Image, exif_dict) -> Image.Image:
    orientation = None
    if exif_dict:
        # exif_dict is a PIL Exif object; map to tag names
        oriented = {}
        for tag_id, value in exif_dict.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            oriented[str(tag)] = value
        orientation = oriented.get("Orientation")
    if orientation is None:
        return img
    try:
        o = int(orientation)
    except Exception:
        return img
    if o == 1:
        return img
    if o == 2:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if o == 3:
        return img.rotate(180, expand=True)
    if o == 4:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if o == 5:
        return img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
    if o == 6:
        return img.rotate(270, expand=True)
    if o == 7:
        return img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
    if o == 8:
        return img.rotate(90, expand=True)
    return img


def load_image_fix_orientation(path: Path) -> Image.Image:
    img = Image.open(path)
    exif = img.getexif()
    img = apply_exif_orientation(img, exif)
    return img


def choose_common_size(sizes: List[Tuple[int, int]]) -> Tuple[int, int]:
    min_w = min(w for (w, h) in sizes)
    min_h = min(h for (w, h) in sizes)
    return (min_w, min_h)


def srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    # arr in [0,1], sRGB to linear (IEC 61966-2-1)
    a = 0.055
    low = arr <= 0.04045
    high = ~low
    out = np.empty_like(arr, dtype=np.float32)
    out[low] = arr[low] / 12.92
    out[high] = ((arr[high] + a) / (1 + a)) ** 2.4
    return out


def linear_to_srgb(arr: np.ndarray) -> np.ndarray:
    # linear to sRGB
    a = 0.055
    low = arr <= 0.0031308
    high = ~low
    out = np.empty_like(arr, dtype=np.float32)
    out[low] = 12.92 * arr[low]
    out[high] = (1 + a) * (arr[high] ** (1 / 2.4)) - a
    return out


def normalize_folder(input_dir: Path, output_dir: Path, linearize: bool = False) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_files(input_dir)
    if not image_paths:
        raise SystemExit(f"No images found in: {input_dir}")

    # Load and compute sizes
    pil_images: List[Image.Image] = []
    sizes: List[Tuple[int, int]] = []
    for p in image_paths:
        img = load_image_fix_orientation(p)
        if img.mode != "RGB":
            img = img.convert("RGB")
        pil_images.append(img)
        sizes.append(img.size)  # (w,h)

    target_w, target_h = choose_common_size(sizes)

    for p, img in zip(image_paths, pil_images):
        # Resize to common size
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.LANCZOS)

        # To float [0,1]
        arr = np.asarray(img).astype(np.float32) / 255.0

        # Optional: linearize for mathematically correct merges down the line
        if linearize:
            arr = srgb_to_linear(arr)

        # Save normalized image as 8-bit PNG (keeps orientation + size normalization)
        save_arr = np.clip(arr, 0.0, 1.0)
        if linearize:
            # For storage as viewable PNG, convert back to sRGB
            save_arr = linear_to_srgb(save_arr)
        save_u8 = (save_arr * 255.0 + 0.5).astype(np.uint8)

        out_name = p.stem + ".png"  # normalized copy
        out_path = output_dir / out_name
        Image.fromarray(save_u8, mode="RGB").save(out_path, format="PNG", optimize=True)
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Normalize bracket images (size, RGB, float[0,1])")
    parser.add_argument("--input", required=True, help="Input folder containing a single bracket set")
    parser.add_argument("--output", required=True, help="Output folder for normalized images")
    parser.add_argument("--linearize", action="store_true", help="Convert to linear light during processing (stored back as sRGB)")
    args = parser.parse_args()

    normalize_folder(Path(args.input), Path(args.output), linearize=bool(args.linearize))


if __name__ == "__main__":
    main()

