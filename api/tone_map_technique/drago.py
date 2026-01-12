from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def tonemap_drago_linear_bgr(
	hdr_bgr: np.ndarray,
	gamma: float = 2.2,
	saturation: float = 1.0,
	bias: float = 0.85,
) -> np.ndarray:
	"""
	OpenCV Drago tone mapping on float32 HDR (BGR). Returns sRGB-like [0,1] in RGB order.
	"""
	if hdr_bgr.dtype != np.float32:
		hdr_bgr = hdr_bgr.astype(np.float32)
	tonemap = cv2.createTonemapDrago(gamma=float(gamma), saturation=float(saturation), bias=float(bias))
	ldr_bgr = np.clip(tonemap.process(hdr_bgr), 0.0, 1.0)  # float32 [0,1]
	ldr_rgb = cv2.cvtColor(ldr_bgr, cv2.COLOR_BGR2RGB)
	return ldr_rgb.astype(np.float32)


def tonemap_drago_file(hdr_path: Path, out_png_path: Path, gamma: float = 2.2, saturation: float = 1.0, bias: float = 0.85) -> str:
	img_bgr = cv2.imread(str(hdr_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
	if img_bgr is None:
		raise RuntimeError(f"Failed to read HDR: {hdr_path}")
	ldr_rgb = tonemap_drago_linear_bgr(img_bgr, gamma=gamma, saturation=saturation, bias=bias)
	u8 = (np.clip(ldr_rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
	out_png_path.parent.mkdir(parents=True, exist_ok=True)
	Image.fromarray(u8, mode="RGB").save(str(out_png_path), format="PNG", optimize=True)
	return str(out_png_path)


