from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
from api.services.image_utils import apply_exif_orientation, choose_common_size, srgb_to_linear, linear_to_srgb


def normalize_set(saved_paths: List[Path], output_dir: Path, linearize: bool = False) -> List[str]:
	output_dir.mkdir(parents=True, exist_ok=True)
	images = []
	sizes = []
	for p in saved_paths:
		img = Image.open(p)
		img = apply_exif_orientation(img, img.getexif())
		if img.mode != "RGB":
			img = img.convert("RGB")
		images.append(img)
		sizes.append(img.size)
	target_w, target_h = choose_common_size(sizes)
	out_paths = []
	for p, img in zip(saved_paths, images):
		if img.size != (target_w, target_h):
			img = img.resize((target_w, target_h), Image.LANCZOS)
		arr = np.asarray(img).astype(np.float32) / 255.0
		if linearize:
			arr = srgb_to_linear(arr)
			save_arr = linear_to_srgb(arr)
		else:
			save_arr = arr
		save_u8 = (np.clip(save_arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
		out_path = output_dir / (p.stem + ".png")
		Image.fromarray(save_u8, mode="RGB").save(out_path, format="PNG", optimize=True)
		out_paths.append(str(out_path))
	return out_paths

