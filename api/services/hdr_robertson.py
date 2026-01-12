from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image
from api.services.image_utils import linear_to_srgb

# Enable EXR if available
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def _read_times(metadata_path: Path, ordered_filenames: List[str]) -> np.ndarray:
	data = json.loads(metadata_path.read_text(encoding="utf-8"))
	name_to_t: Dict[str, float] = {}
	for r in data.get("images", []):
		fn = str(r.get("filename", ""))
		t = r.get("exposure_time_s")
		if isinstance(t, (int, float)) and t and t > 0:
			name_to_t[fn] = float(t)
	times: List[float] = []
	miss = False
	for fn in ordered_filenames:
		if fn in name_to_t:
			times.append(name_to_t[fn])
		else:
			miss = True
			times.append(0.0)
	if miss or not times or sum(times) == 0.0:
		# fallback relative times by rank
		n = len(ordered_filenames)
		center = (n - 1) / 2.0
		times = [2.0 ** (i - center) for i in range(n)]
	return np.array(times, dtype=np.float32)


def _clipping_percent(img_bgr_u8: np.ndarray) -> Dict[str, float]:
	img = img_bgr_u8.astype(np.float32) / 255.0
	y = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return {
		"bright": float(np.mean(y >= 0.98) * 100.0),
		"dark": float(np.mean(y <= 0.02) * 100.0),
	}


def _align_ldr(norm_png_paths: List[Path], transforms_path: Path) -> List[np.ndarray]:
	if transforms_path.exists():
		tdata = json.loads(transforms_path.read_text(encoding="utf-8"))
		f2s = {str(fr.get("filename", "")): (int(fr.get("dx", 0)), int(fr.get("dy", 0))) for fr in tdata.get("frames", [])}
	else:
		f2s = {}
	out: List[np.ndarray] = []
	for p in norm_png_paths:
		img = cv2.imread(str(p), cv2.IMREAD_COLOR)
		if img is None:
			raise RuntimeError(f"Failed to read normalized PNG: {p}")
		dx, dy = f2s.get(p.name, (0, 0))
		if dx != 0 or dy != 0:
			h, w = img.shape[:2]
			M = np.array([[1, 0, float(dx)], [0, 1, float(dy)]], dtype=np.float32)
			img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
		out.append(img)
	return out


def merge_robertson(job_id: str, norm_png_paths: List[Path], metadata_path: Path, transforms_path: Path, out_dir: Path) -> Dict[str, str]:
	out_dir.mkdir(parents=True, exist_ok=True)
	ordered_filenames = [p.name for p in norm_png_paths]
	ldr_list = _align_ldr(norm_png_paths, transforms_path)

	# Clip screening (relaxed): prefer keeping all; if we must reduce, keep the least-clipped frames,
	# but always ensure at least 3 frames survive.
	per_image_clip = []
	for p, img in zip(norm_png_paths, ldr_list):
		clip = _clipping_percent(img)
		per_image_clip.append((p.name, clip, img))
	# Sort by "badness" = bright+dark (lower is better)
	per_image_clip.sort(key=lambda x: (x[1]["bright"] + x[1]["dark"]))
	# Keep all by default
	keep = per_image_clip
	# If more than 5 exposures and some are extreme (>40% bright or >20% dark), drop only the worst until 5 remain
	while len(keep) > 5 and (keep[-1][1]["bright"] > 40.0 or keep[-1][1]["dark"] > 20.0):
		keep.pop()
	# Ensure at least 3 frames
	if len(keep) < 3:
		# fall back to the best 3
		keep = per_image_clip[:3]
	keep_names = [k[0] for k in keep]
	keep_imgs = [k[2] for k in keep]

	times = _read_times(metadata_path, keep_names)

	# Merge using Robertson (robust for JPEGs)
	merger = cv2.createMergeRobertson()
	hdr = merger.process(keep_imgs, times)  # float32 linear (BGR)

	# --- HDR pre-scale using log-average luminance to place mids before tone mapping ---
	eps = 1e-6
	y_hdr = (0.2126 * hdr[..., 2] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 0]).astype(np.float32)
	lum = np.clip(y_hdr, eps, None)
	log_avg = float(np.exp(np.mean(np.log(lum))))
	target_mid = 0.25  # slightly brighter mid for real-estate/interiors
	scale = (target_mid / max(log_avg, eps))
	hdr = np.clip(hdr * scale, 0.0, None)

	# Save HDR EXR if possible else HDR
	exr_path = out_dir / "radiance.exr"
	hdr_path = None
	try:
		if cv2.imwrite(str(exr_path), hdr):
			hdr_path = exr_path
	except Exception:
		hdr_path = None
	if hdr_path is None:
		hdr_path = out_dir / "radiance.hdr"
		ok = cv2.imwrite(str(hdr_path), hdr)
		if not ok:
			raise RuntimeError("Failed to write HDR output for Robertson merge.")

	# Luminance-only Drago tone mapping (removes pink cast):
	# 1) Compute linear RGB → luminance Y
	# 2) Tone-map Y only
	# 3) Scale RGB by Y_tm / Y to preserve colours
	hdr_rgb = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB).astype(np.float32)
	eps = 1e-6
	max_lin = max(float(hdr_rgb.max()), eps)
	hdr_n = hdr_rgb / max_lin  # normalize to [0,1] for tonemapper stability
	Y = (0.2126 * hdr_n[..., 0] + 0.7152 * hdr_n[..., 1] + 0.0722 * hdr_n[..., 2]).astype(np.float32)
	Y3 = np.dstack([Y, Y, Y])
	tm = cv2.createTonemapDrago(gamma=2.2, saturation=1.0, bias=0.75)
	Yt3 = tm.process(Y3)  # [0,1], 3ch
	Yt = np.clip(Yt3[..., 0], 0.0, 1.0)
	scale = Yt / np.clip(Y, eps, None)
	out_lin = np.clip(hdr_n * scale[..., None], 0.0, 1.0)
	out_srgb = np.clip(linear_to_srgb(out_lin), 0.0, 1.0)
	out_srgb = np.nan_to_num(out_srgb, nan=0.0, posinf=1.0, neginf=0.0)

	u8 = (out_srgb * 255.0 + 0.5).astype(np.uint8)
	tone_dir = Path("api/tonemapped_robertson") / job_id
	tone_dir.mkdir(parents=True, exist_ok=True)
	tonemapped_path = tone_dir / "tonemapped.png"
	Image.fromarray(u8, mode="RGB").save(str(tonemapped_path), format="PNG", optimize=True)

	# Simple metrics
	y = 0.2126 * hdr[..., 2] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 0]
	q5, q50, q95 = np.percentile(y, [5, 50, 95])
	dr95 = float(q95 / max(q5, 1e-3))
	metrics = {
		"num_inputs_used": len(keep_imgs),
		"p5": float(q5),
		"p50": float(q50),
		"p95": float(q95),
		"dynamic_range_estimate_p95_p5": dr95,
		"per_image_clip": [{ "file": n, "bright": c["bright"], "dark": c["dark"] } for (n, c, _) in keep],
	}
	with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)

	return {"hdr": str(hdr_path), "tonemapped": str(tonemapped_path), "metrics": str(out_dir / "metrics.json")}

