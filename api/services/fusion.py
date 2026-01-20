from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from api.services.image_utils import linear_to_srgb

def _gentle_post_srgb(
	img_srgb01: np.ndarray,
	sat_gain: float = 1.00,
	warmth: float = 0.0,
	wb_strength: float = 0.60,
	wb_mode: str = "highlights",
	wb_highlight_quantile: float = 0.92,
	wb_max_sat: float = 0.22,
	cool_bias: float = 0.0,
	clahe_clip: float = 2.0,
	clahe_grid: Tuple[int, int] = (8, 8),
	tone_shadow_gamma: float = 0.78,
	tone_highlight_gamma: float = 0.78,
	tone_pivot: float = 0.78,
	sky_sat_boost: float = 1.0,
	sky_hue_min: int = 80,
	sky_hue_max: int = 140,
	sky_s_min: float = 0.05,
	sky_v_min: float = 0.55,
	sharpen_amount: float = 0.0,
	sharpen_radius: float = 1.2,
	sharpen_threshold: float = 0.008,
) -> np.ndarray:
	"""
	Gentle enhancement on an sRGB float image in [0,1].
	- Partial Gray‑world WB (configurable strength)
	- Mild vibrance (boost low‑sat regions more than high‑sat)
	- Optional subtle warm tint (toward yellow)
	- Mild local contrast (CLAHE on luminance)
	"""
	img = np.clip(img_srgb01.astype(np.float32), 0.0, 1.0)
	# WB (partial). Use highlight-based WB by default to keep interior whites clean
	# without being biased by large colored regions (e.g., blue sky through windows).
	wb_strength = float(np.clip(wb_strength, 0.0, 1.0))
	mode = (wb_mode or "gray_world").strip().lower()
	if mode in ("highlights", "highlight", "white_patch", "white"):
		Y = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)
		# Prefer bright *neutral-ish* pixels for WB so blue sky / warm wood doesn't bias WB.
		# Saturation proxy in sRGB: (max-min)/max
		mx = np.max(img, axis=2)
		mn = np.min(img, axis=2)
		S = np.divide(mx - mn, np.maximum(mx, 1e-6))
		sat_max = float(np.clip(wb_max_sat, 0.05, 0.60))
		q = float(np.clip(wb_highlight_quantile, 0.70, 0.995))
		th = float(np.quantile(Y, q))
		mask = (Y >= th) & (S <= sat_max)
		if np.any(mask):
			mean_rgb = np.mean(img[mask].reshape(-1, 3), axis=0) + 1e-6
		else:
			mean_rgb = np.mean(img.reshape(-1, 3), axis=0) + 1e-6
	else:
		mean_rgb = np.mean(img.reshape(-1, 3), axis=0) + 1e-6
	scale = (np.mean(mean_rgb) / mean_rgb).astype(np.float32)
	scale_mix = (1.0 + wb_strength * (scale - 1.0)).astype(np.float32)
	wb = np.clip(img * scale_mix, 0.0, 1.0)

	# Explicit neutralization (tiny cool bias) after WB:
	# negative values warm, positive values cool (reduce R / slightly boost B).
	cb = float(np.clip(cool_bias, -0.10, 0.10))
	if cb != 0.0:
		cool_vec = np.array([1.0 - cb, 1.0, 1.0 + cb], dtype=np.float32)[np.newaxis, np.newaxis, :]
		wb = np.clip(wb * cool_vec, 0.0, 1.0)

	# Mild vibrance in HSV (boost low‑sat regions more). Skip if sat_gain is 1.0.
	if float(sat_gain) == 1.0:
		wb_sat = wb
	else:
		hsv = cv2.cvtColor((wb * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
		s = hsv[..., 1]
		vibrance = 1.0 + (sat_gain - 1.0) * (1.0 - (s / 255.0))
		hsv[..., 1] = np.clip(s * vibrance, 0, 255)
		wb_sat = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
	# Optional subtle warm tint (increase R/G, slightly reduce B)
	wr = 1.0 + warmth
	wg = 1.0 + 0.5 * warmth
	wb_mul = 1.0 - warmth
	warm_vec = np.array([wr, wg, wb_mul], dtype=np.float32)[np.newaxis, np.newaxis, :]
	wb_warm = np.clip(wb_sat * warm_vec, 0.0, 1.0)
	# CLAHE on luminance (optional). For "clean white" looks, lower clip or disable.
	out = wb_warm
	if float(clahe_clip) > 0.0:
		L = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]).astype(np.float32)
		L_u8 = np.clip(L * 255.0 + 0.5, 0, 255).astype(np.uint8)
		clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=clahe_grid)
		L2 = clahe.apply(L_u8).astype(np.float32) / 255.0
		ratio = np.divide(L2, np.maximum(L, 1e-6))
		out = np.clip(out * ratio[..., np.newaxis], 0.0, 1.0)

	# Tone curve to "whiten"/brighten interiors:
	# - Lift shadows/midtones (gamma < 1)
	# - Compress highlights (gamma < 1 on (1-L) distance)
	# Applied on luminance, then re-scaled to RGB to preserve hues.
	tone_pivot_f = float(np.clip(tone_pivot, 0.50, 0.98))
	sg = float(np.clip(tone_shadow_gamma, 0.50, 1.20))
	hg = float(np.clip(tone_highlight_gamma, 0.40, 1.00))
	L3 = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]).astype(np.float32)
	L3 = np.clip(L3, 0.0, 1.0)
	# piecewise gamma around pivot, with continuity at pivot
	shadow = np.power(np.maximum(L3, 1e-6) / tone_pivot_f, sg) * tone_pivot_f
	highlight = 1.0 - np.power(np.maximum((1.0 - L3) / np.maximum(1e-6, 1.0 - tone_pivot_f), 1e-6), hg) * (1.0 - tone_pivot_f)
	L_tone = np.where(L3 <= tone_pivot_f, shadow, highlight).astype(np.float32)
	ratio2 = np.divide(L_tone, np.maximum(L3, 1e-6))
	# Critical: never *brighten* highlights (this is what washes sky to white).
	# For pixels above the pivot, only allow compression (ratio <= 1).
	ratio2 = np.where(L3 > tone_pivot_f, np.minimum(ratio2, 1.0), ratio2)
	out = np.clip(out * ratio2[..., np.newaxis], 0.0, 1.0)

	# Selective color boost for outdoor/sky regions (keeps interior whites clean).
	# This helps restore blue sky saturation after highlight rolloff/toning.
	sb = float(np.clip(sky_sat_boost, 1.0, 2.0))
	if sb > 1.0:
		hsv = cv2.cvtColor((out * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
		h = hsv[..., 0]  # 0..179
		s = hsv[..., 1]  # 0..255
		v = hsv[..., 2]  # 0..255
		hmin = int(np.clip(int(sky_hue_min), 0, 179))
		hmax = int(np.clip(int(sky_hue_max), 0, 179))
		smin = float(np.clip(sky_s_min, 0.0, 1.0)) * 255.0
		vmin = float(np.clip(sky_v_min, 0.0, 1.0)) * 255.0
		# Only boost fairly bright pixels likely to be sky.
		# If the sky is pale (low saturation), hue can be noisy, so also allow a "blue-dominant"
		# test in RGB space.
		mask_hsv = (v >= vmin) & (s >= smin) & (h >= hmin) & (h <= hmax)
		# blue-dominant mask in RGB
		b = out[..., 2]
		g = out[..., 1]
		r = out[..., 0]
		mask_blue = (v >= vmin) & (b > g + 0.03) & (b > r + 0.03)
		mask = mask_hsv | mask_blue
		if np.any(mask):
			s[mask] = np.clip(s[mask] * sb, 0.0, 255.0)
			hsv[..., 1] = s
			out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

	# Subtle sharpening (luminance-only unsharp mask) to add crispness without color halos.
	amt = float(np.clip(sharpen_amount, 0.0, 1.0))
	if amt > 0.0:
		rad = float(np.clip(sharpen_radius, 0.2, 5.0))
		th = float(np.clip(sharpen_threshold, 0.0, 0.05))
		Ls = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]).astype(np.float32)
		Ls_blur = cv2.GaussianBlur(Ls, ksize=(0, 0), sigmaX=rad).astype(np.float32)
		detail = (Ls - Ls_blur).astype(np.float32)
		if th > 0.0:
			mask = (np.abs(detail) > th).astype(np.float32)
		else:
			mask = np.ones_like(detail, dtype=np.float32)
		Ls_sharp = np.clip(Ls + amt * detail * mask, 0.0, 1.0).astype(np.float32)
		ratio_s = np.divide(Ls_sharp, np.maximum(Ls, 1e-6))
		out = np.clip(out * ratio_s[..., np.newaxis], 0.0, 1.0)

	return out


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
	# Apply bias (vectorized)
	boost = np.float32(1.0 + float(k))
	for i in range(len(weights)):
		mask = bright_mask & (k_dark == i)
		if np.any(mask):
			weights[i][mask] *= boost
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


def exposure_fusion_linear(
	images_linear: List[np.ndarray],
	alpha: float = 1.4,
	beta: float = 0.8,
	gamma: float = 0.95,
	levels: int = 8,
	mu: float = 0.30,
	sigma: float = 0.45,
	highlight_thresh: float = 0.95,
	highlight_k: float = 0.18,
	exposure_gain: float = 1.14,
	rolloff_knee: float = 0.78,
	rolloff_strength: float = 0.55,
) -> np.ndarray:
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
	# Small global gain to lift midtones after fusion (linear domain).
	# Keep modest to avoid washing highlights; highlight bias should still protect windows.
	if exposure_gain != 1.0:
		fused = fused * np.float32(exposure_gain)

	# Highlight rolloff in linear space to prevent clipping (keeps blue sky from washing out).
	# Apply on luminance, then scale RGB -> preserves hue/saturation.
	knee = float(np.clip(rolloff_knee, 0.50, 0.95))
	strength = float(np.clip(rolloff_strength, 0.05, 2.0))
	Y = (0.2126 * fused[..., 0] + 0.7152 * fused[..., 1] + 0.0722 * fused[..., 2]).astype(np.float32)
	Y = np.maximum(Y, 0.0)
	d = np.maximum(Y - knee, 0.0)
	den = 1.0 + d / (np.maximum(1e-6, (1.0 - knee) * strength))
	Y2 = np.where(Y <= knee, Y, 1.0 - (1.0 - knee) / den).astype(np.float32)
	ratio_h = np.divide(Y2, np.maximum(Y, 1e-6))
	fused = fused * ratio_h[..., np.newaxis]

	# guard numerics and clamp
	fused = np.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
	return np.clip(fused, 0.0, 1.0).astype(np.float32)


def run_exposure_fusion_from_aligned(
	linear_npy_paths: List[Path],
	out_path: Path,
	levels: int = 8,
	alpha: float = 1.4,
	beta: float = 0.8,
	gamma: float = 0.95,
	mu: float = 0.30,
	sigma: float = 0.45,
	highlight_thresh: float = 0.93,
	highlight_k: float = 0.35,
	exposure_gain: float = 1.16,
	rolloff_knee: float = 0.75,
	rolloff_strength: float = 0.55,
	post_sat_gain: float = 1.00,
	post_warmth: float = 0.0,
	post_wb_strength: float = 1.0,
	post_wb_mode: str = "highlights",
	post_wb_highlight_quantile: float = 0.92,
	post_wb_max_sat: float = 0.22,
	post_cool_bias: float = 0.02,
	post_clahe_clip: float = 0.8,
	post_clahe_grid: Tuple[int, int] = (8, 8),
	post_tone_shadow_gamma: float = 0.72,
	post_tone_highlight_gamma: float = 0.52,
	post_tone_pivot: float = 0.74,
	post_sky_sat_boost: float = 1.35,
	post_sky_hue_min: int = 80,
	post_sky_hue_max: int = 140,
	post_sky_s_min: float = 0.05,
	post_sky_v_min: float = 0.55,
	post_sharpen_amount: float = 0.25,
	post_sharpen_radius: float = 1.2,
	post_sharpen_threshold: float = 0.008,
) -> str:
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
		exposure_gain=exposure_gain,
		rolloff_knee=rolloff_knee,
		rolloff_strength=rolloff_strength,
	)
	# save as sRGB PNG with gentle post‑processing
	fused_srgb = np.clip(linear_to_srgb(fused_linear), 0.0, 1.0).astype(np.float32)
	fused_srgb = _gentle_post_srgb(
		fused_srgb,
		sat_gain=post_sat_gain,
		warmth=post_warmth,
		wb_strength=post_wb_strength,
		wb_mode=post_wb_mode,
		wb_highlight_quantile=post_wb_highlight_quantile,
		wb_max_sat=post_wb_max_sat,
		cool_bias=post_cool_bias,
		clahe_clip=post_clahe_clip,
		clahe_grid=post_clahe_grid,
		tone_shadow_gamma=post_tone_shadow_gamma,
		tone_highlight_gamma=post_tone_highlight_gamma,
		tone_pivot=post_tone_pivot,
		sky_sat_boost=post_sky_sat_boost,
		sky_hue_min=post_sky_hue_min,
		sky_hue_max=post_sky_hue_max,
		sky_s_min=post_sky_s_min,
		sky_v_min=post_sky_v_min,
		sharpen_amount=post_sharpen_amount,
		sharpen_radius=post_sharpen_radius,
		sharpen_threshold=post_sharpen_threshold,
	)
	u8 = (np.clip(fused_srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	Image.fromarray(u8, mode="RGB").save(str(out_path), format="PNG", optimize=True)
	return str(out_path)


