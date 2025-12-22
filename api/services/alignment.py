from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class ShiftResult:
	dx: int
	dy: int
	level_costs: List[int]
	overlap_ratio: float


def _to_gray_linear(rgb: np.ndarray) -> np.ndarray:
	"""
	Convert linear-light RGB float32 image [H,W,3] in [0,1] to luminance gray [H,W] in [0,1].
	Uses Rec.709 coefficients.
	"""
	if rgb.ndim != 3 or rgb.shape[2] != 3:
		raise ValueError("Expected HxWx3 linear RGB array")
	r = rgb[..., 0].astype(np.float32)
	g = rgb[..., 1].astype(np.float32)
	b = rgb[..., 2].astype(np.float32)
	return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _build_mtb(gray: np.ndarray, exclude_band: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Build Median Threshold Bitmap (MTB) and exclusion mask from a gray image in [0,1].
	Returns (bitmap, exclusion) as uint8 arrays with values {0,1}.
	"""
	median = float(np.median(gray))
	bitmap = (gray > median).astype(np.uint8)
	excl = (np.abs(gray - median) < float(exclude_band)).astype(np.uint8)
	return bitmap, excl


def _pyramid_from_gray(gray: np.ndarray, max_levels: int = 5, min_size: int = 32, exclude_band: float = 0.02) -> List[Tuple[np.ndarray, np.ndarray]]:
	"""
	Build pyramid list of (bitmap, exclusion) from finest (level 0) to coarsest (last).
	"""
	levels: List[Tuple[np.ndarray, np.ndarray]] = []
	g = gray.astype(np.float32)
	while True:
		b, e = _build_mtb(g, exclude_band=exclude_band)
		levels.append((b, e))
		h, w = g.shape[:2]
		if len(levels) >= max_levels or min(h, w) // 2 < min_size:
			break
		# Downsample using pyrDown (Gaussian + decimate)
		g = cv2.pyrDown(g)
	return levels


def _mismatch_cost(b_ref: np.ndarray, e_ref: np.ndarray, b_mov: np.ndarray, e_mov: np.ndarray, dx: int, dy: int) -> Tuple[int, float]:
	"""
	Compute XOR mismatch count between reference and shifted moving bitmap,
	ignoring excluded pixels. Also returns overlap ratio in [0,1].
	All bitmaps/masks are uint8 with {0,1}.
	"""
	h, w = b_ref.shape[:2]
	xr0 = max(0, dx)
	yr0 = max(0, dy)
	xm0 = max(0, -dx)
	ym0 = max(0, -dy)
	width = min(w - xr0, w - xm0)
	height = min(h - yr0, h - ym0)
	if width <= 0 or height <= 0:
		return 10**12, 0.0
	br = b_ref[yr0:yr0 + height, xr0:xr0 + width]
	er = e_ref[yr0:yr0 + height, xr0:xr0 + width]
	bm = b_mov[ym0:ym0 + height, xm0:xm0 + width]
	em = e_mov[ym0:ym0 + height, xm0:xm0 + width]
	# mask out uncertain pixels
	mask = 1 - np.minimum(1, er + em)  # 1 where both not excluded
	xor = cv2.bitwise_xor(br, bm)
	mism = int(cv2.countNonZero(xor & mask))
	overlap = float(width * height) / float(w * h)
	return mism, overlap


def estimate_translation_mtb(
	ref_gray: np.ndarray,
	mov_gray: np.ndarray,
	max_levels: int = 5,
	base_radius: int = 4,
	exclude_band: float = 0.02,
	min_size: int = 32,
) :  # -> ShiftResult
	"""
	Estimate global integer translation (dx, dy) of mov_gray onto ref_gray using MTB pyramid search.
	Returns ShiftResult with per-level costs.
	"""
	ref_pyr = _pyramid_from_gray(ref_gray, max_levels=max_levels, min_size=min_size, exclude_band=exclude_band)
	mov_pyr = _pyramid_from_gray(mov_gray, max_levels=max_levels, min_size=min_size, exclude_band=exclude_band)
	L = min(len(ref_pyr), len(mov_pyr))
	ref_pyr = ref_pyr[:L]
	mov_pyr = mov_pyr[:L]

	ox = 0
	oy = 0
	level_costs: List[int] = []
	overlap_final = 0.0

	for li in range(L - 1, -1, -1):  # coarse -> fine
		b_ref, e_ref = ref_pyr[li]
		b_mov, e_mov = mov_pyr[li]

		# scale from coarser to current level (skip for coarsest)
		if li != L - 1:
			ox *= 2
			oy *= 2

		# shrinking search radius as we go finer
		shrink = max(1, 2 ** (L - 1 - li))
		radius = max(1, int(round(base_radius / shrink)))

		best_cost = 10**12
		best_dx = ox
		best_dy = oy
		for dy in range(oy - radius, oy + radius + 1):
			for dx in range(ox - radius, ox + radius + 1):
				cost, overlap = _mismatch_cost(b_ref, e_ref, b_mov, e_mov, dx, dy)
				if cost < best_cost:
					best_cost = cost
					best_dx = dx
					best_dy = dy
		ox, oy = best_dx, best_dy
		level_costs.append(best_cost)
		if li == 0:
			# compute final overlap at full res
			_, overlap_final = _mismatch_cost(b_ref, e_ref, b_mov, e_mov, ox, oy)

	return ShiftResult(dx=int(ox), dy=int(oy), level_costs=level_costs, overlap_ratio=overlap_final)


def _warp_linear_rgb(arr: np.ndarray, dx: int, dy: int) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Warp linear-light RGB array by integer translation (dx, dy) using bilinear interpolation.
	Returns (warped_arr, valid_mask).
	"""
	h, w = arr.shape[:2]
	M = np.array([[1, 0, float(dx)], [0, 1, float(dy)]], dtype=np.float32)
	warped = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
	# Build valid mask (ones warped the same way, then clamp to {0,1})
	ones = np.ones((h, w), dtype=np.uint8) * 255
	mask = cv2.warpAffine(ones, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
	mask = (mask > 0).astype(np.uint8)
	return warped, mask


def align_set(job_id: str, npy_paths: List[Path], ref_index: int, out_dir: Path, save_png: bool = False) -> Dict[str, object]:
	"""
	Align a set of linear-light RGB arrays (0..1 float32) using MTB translation alignment.
	Saves aligned arrays as <stem>_aligned.npy and a JSON transform file.
	Returns dict with keys: aligned_paths (List[str]), transforms (str).
	"""
	out_dir.mkdir(parents=True, exist_ok=True)

	# Load all arrays (small brackets; if very large sets, stream instead)
	arrs: List[np.ndarray] = [np.load(str(p)) for p in npy_paths]
	grays: List[np.ndarray] = [_to_gray_linear(a) for a in arrs]
	ref_gray = grays[ref_index]
	ref_name = npy_paths[ref_index].name

	transforms: Dict[str, object] = {
		"job_id": job_id,
		"reference_index": int(ref_index),
		"reference": ref_name,
		"frames": [],
	}

	aligned_paths: List[str] = []

	for idx, (p, arr, gray) in enumerate(zip(npy_paths, arrs, grays)):
		stem = Path(p).stem.replace("_linear", "")
		if idx == ref_index:
			aligned = arr.copy()
			mask = np.ones(arr.shape[:2], dtype=np.uint8)
			shift = ShiftResult(dx=0, dy=0, level_costs=[0], overlap_ratio=1.0)
		else:
			shift = estimate_translation_mtb(ref_gray, gray)
			aligned, mask = _warp_linear_rgb(arr, shift.dx, shift.dy)

		aligned_path = out_dir / f"{stem}_aligned.npy"
		np.save(str(aligned_path), aligned.astype(np.float32))
		aligned_paths.append(str(aligned_path))

		frame_info = {
			"index": int(idx),
			"filename": p.name,
			"dx": int(shift.dx),
			"dy": int(shift.dy),
			"pyramid_costs": [int(c) for c in shift.level_costs],
			"overlap_ratio": float(shift.overlap_ratio),
		}
		transforms["frames"].append(frame_info)  # type: ignore[index]

		# Optionally save a viewable PNG for quick manual check (off by default)
		if save_png:
			# Convert linear -> sRGB for display
			from .image_utils import linear_to_srgb  # local import to avoid cycles
			display = np.clip(linear_to_srgb(aligned), 0.0, 1.0)
			u8 = (display * 255.0 + 0.5).astype(np.uint8)
			cv2.imwrite(str(out_dir / f"{stem}_aligned.png"), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))

	transforms_path = out_dir / "transforms.json"
	with transforms_path.open("w", encoding="utf-8") as f:
		json.dump(transforms, f, indent=2)

	return {
		"aligned_paths": aligned_paths,
		"transforms": str(transforms_path),
	}


