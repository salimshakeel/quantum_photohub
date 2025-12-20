from __future__ import annotations
from typing import List, Tuple
import numpy as np
from PIL import ExifTags, Image


def apply_exif_orientation(img: Image.Image, exif) -> Image.Image:
	orientation = None
	if exif:
		tmp = {}
		for tag_id, value in exif.items():
			tag = ExifTags.TAGS.get(tag_id, tag_id)
			tmp[str(tag)] = value
		orientation = tmp.get("Orientation")
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


def choose_common_size(sizes: List[Tuple[int, int]]) -> Tuple[int, int]:
	min_w = min(w for (w, h) in sizes)
	min_h = min(h for (w, h) in sizes)
	return (min_w, min_h)


def srgb_to_linear(arr: np.ndarray) -> np.ndarray:
	a = 0.055
	low = arr <= 0.04045
	high = ~low
	out = np.empty_like(arr, dtype=np.float32)
	out[low] = arr[low] / 12.92
	out[high] = ((arr[high] + a) / (1 + a)) ** 2.4
	return out


def linear_to_srgb(arr: np.ndarray) -> np.ndarray:
	a = 0.055
	low = arr <= 0.0031308
	high = ~low
	out = np.empty_like(arr, dtype=np.float32)
	out[low] = 12.92 * arr[low]
	out[high] = (1 + a) * (arr[high] ** (1 / 2.4)) - a
	return out

