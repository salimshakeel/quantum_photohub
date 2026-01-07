from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from api.services.image_utils import linear_to_srgb


def _compute_log_average_luminance(
	lum: np.ndarray,
	epsilon: float = 1e-6,
	exclude_low_pct: float = 0.01,
	exclude_high_pct: float = 0.02,
) -> float:
	"""
	Log-average luminance with optional percentile clipping to reduce
	influence from extreme shadows/highlights.
	"""
	l = lum.astype(np.float32).reshape(-1)
	if 0.0 < exclude_low_pct < 0.5:
		low = np.percentile(l, exclude_low_pct * 100.0)
	else:
		low = None
	if 0.0 < exclude_high_pct < 0.5:
		high = np.percentile(l, 100.0 * (1.0 - exclude_high_pct))
	else:
		high = None
	if low is not None and high is not None and high > low:
		mask = (l >= low) & (l <= high)
		l = l[mask] if np.any(mask) else l
	lum_clamped = np.clip(l, epsilon, None)
	return float(np.exp(np.mean(np.log(lum_clamped))))


def tonemap_reinhard_linear(
	hdr_linear_rgb: np.ndarray,
	key: float = 0.45,
	white: Optional[float] = 3.0,
	gamma: float = 2.2,
	exclude_low_pct: float = 0.01,
	exclude_high_pct: float = 0.02,
	contrast: float = 1.05,
) -> np.ndarray:
	"""
	Apply Reinhard global tone mapping to a linear-light HDR RGB image.
	Returns an sRGB image in [0,1].
	"""
	if hdr_linear_rgb.dtype != np.float32:
		hdr = hdr_linear_rgb.astype(np.float32)
	else:
		hdr = hdr_linear_rgb

	# Luminance in linear light (Rec.709 coefficients)
	L = 0.2126 * hdr[..., 0] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 2]
	L_bar = _compute_log_average_luminance(L, exclude_low_pct=exclude_low_pct, exclude_high_pct=exclude_high_pct)

	# Scale scene by key
	L_s = (key / max(L_bar, 1e-6)) * L

	# Compress highlights
	if white is not None and white > 0:
		L_d = (L_s * (1.0 + (L_s / (white * white)))) / (1.0 + L_s)
	else:
		L_d = L_s / (1.0 + L_s)

	# Re-apply color via luminance ratio (avoid divide-by-zero)
	ratio = np.divide(L_d, np.maximum(L, 1e-6))
	out_linear = hdr * ratio[..., np.newaxis]

	# Convert to sRGB for viewing
	out_srgb = np.clip(linear_to_srgb(np.clip(out_linear, 0.0, 1.0)), 0.0, 1.0)

	# Optional gamma (applied on sRGB; for most workflows leave at 2.2 to match displays)
	if abs(gamma - 2.2) > 1e-3:
		out_srgb = np.clip(out_srgb ** (1.0 / gamma), 0.0, 1.0)

	# Subtle global contrast boost to counter hazy look
	if abs(contrast - 1.0) > 1e-3:
		out_srgb = np.clip((out_srgb - 0.5) * contrast + 0.5, 0.0, 1.0)

	return out_srgb.astype(np.float32)


def tonemap_reinhard_file(
	hdr_path: Path,
	out_png_path: Path,
	key: float = 0.45,
	white: Optional[float] = 3.0,
	gamma: float = 2.2,
	exclude_low_pct: float = 0.01,
	exclude_high_pct: float = 0.02,
	contrast: float = 1.05,
) -> str:
	"""
	Load an HDR file (EXR/HDR) as float32 (BGR from OpenCV), convert to RGB,
	apply Reinhard tone mapping, convert to sRGB, and save PNG.
	Returns the saved path.
	"""
	# Read with OpenCV (ANYDEPTH keeps float; COLOR loads 3 channels)
	img_bgr = cv2.imread(str(hdr_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
	if img_bgr is None:
		raise RuntimeError(f"Failed to read HDR: {hdr_path}")
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

	out_srgb = tonemap_reinhard_linear(
		img_rgb,
		key=key,
		white=white,
		gamma=gamma,
		exclude_low_pct=exclude_low_pct,
		exclude_high_pct=exclude_high_pct,
		contrast=contrast,
	)
	u8 = (np.clip(out_srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
	out_png_path.parent.mkdir(parents=True, exist_ok=True)
	Image.fromarray(u8, mode="RGB").save(str(out_png_path), format="PNG", optimize=True)
	return str(out_png_path)


