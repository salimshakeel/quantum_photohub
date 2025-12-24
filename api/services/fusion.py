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


def _well_exposed_weight(img_rgb: np.ndarray, sigma: float = 0.2) -> np.ndarray:
	# product of per-channel Gaussians around 0.5
	c = np.exp(-0.5 * ((img_rgb - 0.5) ** 2) / (sigma ** 2))
	w = c[..., 0] * c[..., 1] * c[..., 2]
	return w.astype(np.float32) + 1e-12


def _normalize_weights(weights: List[np.ndarray]) -> List[np.ndarray]:
	stack = np.stack(weights, axis=0)  # [N,H,W]
	den = np.sum(stack, axis=0, keepdims=False) + 1e-12
	return [(w / den).astype(np.float32) for w in weights]


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


def exposure_fusion_srgb(images_srgb: List[np.ndarray], alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0, levels: int = 6) -> np.ndarray:
	# weights
	weights: List[np.ndarray] = []
	for img in images_srgb:
		wc = _contrast_weight(img) ** alpha
		ws = _saturation_weight(img) ** beta
		we = _well_exposed_weight(img) ** gamma
		w = (wc * ws * we).astype(np.float32)
		weights.append(w)
	weights = _normalize_weights(weights)

	# pyramids and fuse
	# weights as Gaussian pyramids
	weights_gp = [ _gaussian_pyramid(w, levels) for w in weights ]
	# images as Laplacian pyramids (per channel)
	img_lp = [ _laplacian_pyramid(img, levels) for img in images_srgb ]

	fused_lp: List[np.ndarray] = []
	for lvl in range(levels):
		acc = np.zeros_like(img_lp[0][lvl], dtype=np.float32)
		for k in range(len(images_srgb)):
			w = weights_gp[k][lvl][..., np.newaxis]  # broadcast to 3 channels
			acc += w * img_lp[k][lvl]
		fused_lp.append(acc.astype(np.float32))

	fused = _collapse_laplacian_pyr(fused_lp)
	return np.clip(fused, 0.0, 1.0).astype(np.float32)


def run_exposure_fusion_from_aligned(linear_npy_paths: List[Path], out_path: Path, levels: int = 6, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0) -> str:
	"""
	Load aligned linear arrays (*.npy), convert to sRGB for weighting, fuse, and save PNG at out_path.
	Returns the saved path as string.
	"""
	images_linear: List[np.ndarray] = [np.load(str(p)).astype(np.float32) for p in linear_npy_paths]
	images_srgb: List[np.ndarray] = [np.clip(linear_to_srgb(img), 0.0, 1.0).astype(np.float32) for img in images_linear]
	fused = exposure_fusion_srgb(images_srgb, alpha=alpha, beta=beta, gamma=gamma, levels=levels)
	u8 = (fused * 255.0 + 0.5).astype(np.uint8)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	Image.fromarray(u8, mode="RGB").save(str(out_path), format="PNG", optimize=True)
	return str(out_path)


