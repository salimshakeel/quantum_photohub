from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from api.services.image_utils import linear_to_srgb


def _to_gray(arr_rgb: np.ndarray) -> np.ndarray:
	r = arr_rgb[..., 0].astype(np.float32)
	g = arr_rgb[..., 1].astype(np.float32)
	b = arr_rgb[..., 2].astype(np.float32)
	return 0.299 * r + 0.587 * g + 0.114 * b


def _contrast_weight(img_rgb: np.ndarray) -> np.ndarray:
	gray = _to_gray(img_rgb)
	lap = cv2.Laplacian(gray, ddepth=cv2.CV_32F, ksize=3)
	return np.abs(lap) + 1e-12


def _saturation_weight(img_rgb: np.ndarray) -> np.ndarray:
	# std across channels
	return np.std(img_rgb, axis=2).astype(np.float32) + 1e-12


def _well_exposed_weight(img_rgb: np.ndarray, mu: float = 0.18, sigma: float = 0.3) -> np.ndarray:
	"""
	Per-channel well-exposedness around a center mu (linear middle gray default ~0.18).
	"""
	c = np.exp(-0.5 * ((img_rgb - mu) ** 2) / (sigma ** 2))
	w = c[..., 0] * c[..., 1] * c[..., 2]
	return w.astype(np.float32) + 1e-12


def _normalize_weights(weights: List[np.ndarray]) -> List[np.ndarray]:
	stack = np.stack(weights, axis=0)  # [N,H,W]
	den = np.sum(stack, axis=0, keepdims=False)
	tiny = 1e-6
	# safe inverse with fallback to zero
	inv = np.divide(1.0, den, out=np.zeros_like(den, dtype=np.float32), where=den > tiny)
	norm = [(w * inv).astype(np.float32) for w in weights]
	# if denominator is tiny, fall back to uniform weights
	if len(weights) > 0:
		uniform = np.float32(1.0 / float(len(weights)))
		mask = den <= tiny
		if np.any(mask):
			for i in range(len(norm)):
				norm[i][mask] = uniform
	return norm


def _apply_highlight_bias(weights: List[np.ndarray], images_linear: List[np.ndarray], threshold: float = 0.85, k: float = 0.2) -> List[np.ndarray]:
	"""
	Bias weights toward the darkest exposure in very bright regions (linear space).
	- threshold: luminance threshold in [0,1] to define "bright" areas (e.g., 0.85)
	- k: multiplicative boost for the darkest exposure (e.g., 0.2 -> +20%)
	"""
	if not weights:
		return weights
	# Compute luminance per exposure
	Ys: List[np.ndarray] = [0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2] for im in images_linear]
	Y_stack = np.stack(Ys, axis=0)  # [N,H,W]
	# Bright mask based on median luminance across stack
	bright_mask = (np.median(Y_stack, axis=0) > float(threshold))
	if not np.any(bright_mask):
		return weights
	# Darkest exposure index per pixel
	k_dark = np.argmin(Y_stack, axis=0)  # [H,W] int
	# Apply bias
	for h in range(weights[0].shape[0]):
		for w in range(weights[0].shape[1]):
			if bright_mask[h, w]:
				idx = int(k_dark[h, w])
				weights[idx][h, w] *= (1.0 + float(k))
	return weights


def _gaussian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
	pyr = [img]
	for _ in range(1, levels):
		img = cv2.pyrDown(img)
		pyr.append(img)
	return pyr


def _laplacian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
	gp = _gaussian_pyramid(img, levels)
	lp: List[np.ndarray] = []
	for i in range(levels - 1):
		size = (gp[i].shape[1], gp[i].shape[0])
		up = cv2.pyrUp(gp[i + 1], dstsize=size)
		lp.append((gp[i] - up).astype(np.float32))
	lp.append(gp[-1].astype(np.float32))
	return lp


def _collapse_laplacian_pyr(lp: List[np.ndarray]) -> np.ndarray:
	img = lp[-1]
	for i in range(len(lp) - 2, -1, -1):
		size = (lp[i].shape[1], lp[i].shape[0])
		img = cv2.pyrUp(img, dstsize=size)
		img = (img + lp[i]).astype(np.float32)
	return img


def exposure_fusion_linear(images_linear: List[np.ndarray], alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0, levels: int = 7, mu: float = 0.18, sigma: float = 0.3, highlight_thresh: float = 0.85, highlight_k: float = 0.2) -> np.ndarray:
	"""
	Exposure fusion performed entirely in linear light (weights and blending).
	Returns fused linear RGB image in [0,1].
	"""
	# weights in linear
	weights: List[np.ndarray] = []
	w_floor = 1e-3  # prevent weight collapse in flat/bright regions
	for img in images_linear:
		wc = _contrast_weight(img) ** alpha
		ws = _saturation_weight(img) ** beta
		we = _well_exposed_weight(img, mu=mu, sigma=sigma) ** gamma
		w = (wc * ws * we + w_floor).astype(np.float32)
		weights.append(w)

	# highlight protection bias toward darkest exposure where bright
	weights = _apply_highlight_bias(weights, images_linear, threshold=highlight_thresh, k=highlight_k)
	weights = _normalize_weights(weights)

	# pyramids and fuse
	# weights as Gaussian pyramids
	weights_gp = [ _gaussian_pyramid(w, levels) for w in weights ]
	# images as Laplacian pyramids (per channel)
	img_lp = [ _laplacian_pyramid(img, levels) for img in images_linear ]

	fused_lp: List[np.ndarray] = []
	for lvl in range(levels):
		acc = np.zeros_like(img_lp[0][lvl], dtype=np.float32)
		for i in range(len(images_linear)):
			w = weights_gp[i][lvl][..., np.newaxis]  # broadcast to 3 channels
			acc += w * img_lp[i][lvl]
		fused_lp.append(acc.astype(np.float32))

	fused = _collapse_laplacian_pyr(fused_lp)
	# guard numerics and clamp
	fused = np.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
	return np.clip(fused, 0.0, 1.0).astype(np.float32)


def run_exposure_fusion_from_aligned(linear_npy_paths: List[Path], out_path: Path, levels: int = 7, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0, mu: float = 0.18, sigma: float = 0.3, highlight_thresh: float = 0.85, highlight_k: float = 0.2) -> str:
	"""
	Load aligned linear arrays (*.npy), perform linear-space fusion with highlight protection,
	then convert to sRGB for saving PNG at out_path.
	Returns the saved path as string.
	"""
	images_linear: List[np.ndarray] = [np.load(str(p)).astype(np.float32) for p in linear_npy_paths]
	fused_linear = exposure_fusion_linear(
		images_linear,
		alpha=alpha,
		beta=beta,
		gamma=gamma,
		levels=levels,
		mu=mu,
		sigma=sigma,
		highlight_thresh=highlight_thresh,
		highlight_k=highlight_k,
	)
	# save as sRGB PNG
	fused_srgb = np.clip(linear_to_srgb(fused_linear), 0.0, 1.0).astype(np.float32)
	u8 = (fused_srgb * 255.0 + 0.5).astype(np.uint8)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	Image.fromarray(u8, mode="RGB").save(str(out_path), format="PNG", optimize=True)
	return str(out_path)


