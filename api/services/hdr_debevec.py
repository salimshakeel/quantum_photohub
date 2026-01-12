from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import os
# Try to enable OpenEXR support in OpenCV if available
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from PIL import Image
from api.tone_map_technique.reinhard import tonemap_reinhard_linear
from api.tone_map_technique.drago import tonemap_drago_linear_bgr
from api.services.image_utils import srgb_to_linear


def _read_times_from_metadata(metadata_path: Path, ordered_filenames: List[str]) -> np.ndarray:
	"""
	Read exposure times from metadata.json. If missing, approximate relative times by rank.
	Returns float32 array aligned to ordered_filenames.
	"""
	with metadata_path.open("r", encoding="utf-8") as f:
		meta: Dict[str, Any] = json.load(f)
	records: Dict[str, float] = {}
	for rec in meta.get("images", []):
		name = str(rec.get("filename", ""))
		t = rec.get("exposure_time_s", None)
		if isinstance(t, (int, float)) and t and t > 0:
			records[name] = float(t)

	# build times, fallback if any missing
	times: List[float] = []
	any_missing = False
	for fn in ordered_filenames:
		if fn in records:
			times.append(records[fn])
		else:
			any_missing = True
			times.append(0.0)

	if any_missing or not times or sum(times) == 0.0:
		# fallback: relative times by rank around center (e.g., 0.5,1,2 for 3 images)
		n = len(ordered_filenames)
		center = (n - 1) / 2.0
		times = [2.0 ** (i - center) for i in range(n)]

	return np.array(times, dtype=np.float32)


def _load_ldr_png(path: Path) -> np.ndarray:
	"""
	Load PNG as 8-bit RGB (OpenCV uses BGR by default; we'll convert to BGR for OpenCV calls if needed).
	Returns uint8 HxWx3 in BGR for OpenCV HDR APIs.
	"""
	img = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if img is None:
		raise RuntimeError(f"Failed to read image: {path}")
	# ensure 8-bit
	if img.dtype != np.uint8:
		img = np.clip(img, 0, 255).astype(np.uint8)
	return img


def _apply_integer_shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
	h, w = img.shape[:2]
	M = np.array([[1, 0, float(dx)], [0, 1, float(dy)]], dtype=np.float32)
	return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _read_transforms(transforms_path: Path, ordered_filenames: List[str]) -> List[Tuple[int, int]]:
	with transforms_path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	frames = data.get("frames", [])
	# map filename -> (dx,dy)
	name_to_shift: Dict[str, Tuple[int, int]] = {}
	for fr in frames:
		name_to_shift[str(fr.get("filename", ""))] = (int(fr.get("dx", 0)), int(fr.get("dy", 0)))
	shifts: List[Tuple[int, int]] = []
	for fn in ordered_filenames:
		shifts.append(name_to_shift.get(fn, (0, 0)))
	return shifts


def prepare_aligned_ldr(norm_png_paths: List[Path], transforms_path: Path) -> List[np.ndarray]:
	"""
	Load normalized PNGs and apply integer shifts from transforms.json so the 8-bit stack is aligned.
	Returns list of BGR uint8 images.
	"""
	ordered_filenames = [p.name for p in norm_png_paths]
	shifts = _read_transforms(transforms_path, ordered_filenames) if transforms_path.exists() else [(0, 0)] * len(norm_png_paths)
	out: List[np.ndarray] = []
	for p, (dx, dy) in zip(norm_png_paths, shifts):
		img = _load_ldr_png(p)
		if dx != 0 or dy != 0:
			img = _apply_integer_shift(img, dx, dy)
		out.append(img)
	return out


def merge_debevec(job_id: str, norm_png_paths: List[Path], metadata_path: Path, transforms_path: Path, out_dir: Path) -> Dict[str, str]:
	"""
	Run Debevec-Malik HDR pipeline:
	- align LDR PNGs with stored (dx,dy)
	- calibrate camera response
	- merge to HDR radiance
	Saves HDR as .exr (float32). Returns paths.
	"""
	out_dir.mkdir(parents=True, exist_ok=True)
	ordered_filenames = [p.name for p in norm_png_paths]
	ldr_imgs = prepare_aligned_ldr(norm_png_paths, transforms_path)
	times = _read_times_from_metadata(metadata_path, ordered_filenames)

	calib = cv2.createCalibrateDebevec()
	response = calib.process(ldr_imgs, times)

	merger = cv2.createMergeDebevec()
	hdr = merger.process(ldr_imgs, times, response)  # float32 linear radiance (BGR)

	# Save HDR: prefer EXR; if not supported, fallback to Radiance .hdr
	exr_path = out_dir / "radiance.exr"
	wrote_exr = False
	try:
		wrote_exr = bool(cv2.imwrite(str(exr_path), hdr))
	except Exception:
		wrote_exr = False
	if wrote_exr:
		hdr_path = exr_path
	else:
		hdr_path = out_dir / "radiance.hdr"
		ok = cv2.imwrite(str(hdr_path), hdr)
		if not ok:
			raise RuntimeError("Failed to write HDR output (.exr disabled and .hdr write failed).")

	# Tone-map preview: try Drago (better for strong highlights). If you want Reinhard, swap call.
	srgb_tm = tonemap_drago_linear_bgr(hdr, gamma=2.2, saturation=1.0, bias=0.80)  # returns sRGB [0,1]
	u8 = (np.clip(srgb_tm, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
	# Save tonemapped to a separate folder to avoid confusion with HDR outputs
	tone_dir = Path("api/tonemapped_dm") / job_id
	tone_dir.mkdir(parents=True, exist_ok=True)
	tonemapped_path = tone_dir / "tonemapped.png"
	Image.fromarray(u8, mode="RGB").save(str(tonemapped_path), format="PNG", optimize=True)

	# Minimal metrics (inputs, times)
	metrics = {"num_inputs": len(ldr_imgs), "times": [float(t) for t in times]}
	with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)

	return {
		"hdr": str(hdr_path),
		"tonemapped": str(tonemapped_path),
		"metrics": str(out_dir / "metrics.json"),
	}


