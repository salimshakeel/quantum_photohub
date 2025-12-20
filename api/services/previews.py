from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image

from api.services.image_utils import apply_exif_orientation


def generate_previews(saved_paths: List[Path], preview_dir: Path) -> List[str]:
	preview_dir.mkdir(parents=True, exist_ok=True)
	out = []
	for p in saved_paths:
		img = Image.open(p)
		img = apply_exif_orientation(img, img.getexif())
		if img.mode != "RGB":
			img = img.convert("RGB")
		max_w = 512
		if img.width > max_w:
			r = max_w / float(img.width)
			img = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)
		out_path = preview_dir / (p.stem + ".jpg")
		img.save(out_path, format="JPEG", quality=85, optimize=True)
		out.append(str(out_path))
	return out

