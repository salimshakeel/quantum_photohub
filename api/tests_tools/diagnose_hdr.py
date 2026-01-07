from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


def _luma_from_rgb01(img: np.ndarray) -> np.ndarray:
	# img: float32 [0,1], RGB
	return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)


def check_clipping_normalized(job_id: str) -> List[Dict[str, float]]:
	norm_dir = Path("api/normalized") / job_id
	results: List[Dict[str, float]] = []
	for p in sorted(norm_dir.glob("*.png")):
		im = Image.open(p).convert("RGB")
		arr = np.asarray(im).astype(np.float32) / 255.0
		y = _luma_from_rgb01(arr)
		bright = float(np.mean(y >= 0.98) * 100.0)
		dark = float(np.mean(y <= 0.02) * 100.0)
		results.append({"file": p.name, "bright_clip_percent": round(bright, 3), "dark_clip_percent": round(dark, 3)})
	return results


def check_linearization(job_id: str) -> List[Dict[str, float]]:
	lin_dir = Path("api/linear") / job_id
	out: List[Dict[str, float]] = []
	for p in sorted(lin_dir.glob("*_linear.npy")):
		arr = np.load(p).astype(np.float32)
		y = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)
		q1, q50, q99 = np.percentile(y, [1, 50, 99])
		out.append({
			"file": p.name,
			"p1": float(q1),
			"p50": float(q50),
			"p99": float(q99),
		})
	return out


def check_alignment(job_id: str) -> Dict[str, object]:
	tpath = Path("api/aligned") / job_id / "transforms.json"
	if not tpath.exists():
		return {"error": f"transforms.json not found for {job_id}"}
	data = json.loads(tpath.read_text(encoding="utf-8"))
	frames = data.get("frames", [])
	shifts = [(int(f.get("dx", 0)), int(f.get("dy", 0))) for f in frames]
	max_shift = max((abs(dx) + abs(dy) for dx, dy in shifts), default=0)
	return {
		"reference": data.get("reference"),
		"num_frames": len(frames),
		"shifts": shifts,
		"max_abs_shift": int(max_shift),
	}


def check_hdr(job_id: str) -> Dict[str, object]:
	hdr_path = Path("api/hdr_dm") / job_id / "radiance.exr"
	if not hdr_path.exists():
		hdr_path = Path("api/hdr_dm") / job_id / "radiance.hdr"
	if not hdr_path.exists():
		return {"error": f"HDR file not found for {job_id}"}
	hdr = cv2.imread(str(hdr_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
	if hdr is None:
		return {"error": f"Failed to read HDR {hdr_path}"}
	y = (0.2126 * hdr[..., 2] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 0]).astype(np.float32)
	q1, q50, q99 = np.percentile(y, [1, 50, 99])
	dr = float(q99 / max(q1, 1e-6))
	return {
		"hdr_path": str(hdr_path),
		"p1": float(q1),
		"p50": float(q50),
		"p99": float(q99),
		"dynamic_range_estimate": dr,
		"has_nan": bool(np.isnan(y).any()),
		"has_inf": bool(np.isinf(y).any()),
	}


def run(job_id: str) -> Dict[str, object]:
	report = {
		"job_id": job_id,
		"normalized_clipping": check_clipping_normalized(job_id),
		"linearization_percentiles": check_linearization(job_id),
		"alignment": check_alignment(job_id),
		"hdr": check_hdr(job_id),
	}
	out_dir = Path("api/tests") / job_id
	out_dir.mkdir(parents=True, exist_ok=True)
	with (out_dir / "diagnostics.json").open("w", encoding="utf-8") as f:
		json.dump(report, f, indent=2)
	print(json.dumps(report, indent=2))
	return report


def main():
	parser = argparse.ArgumentParser(description="Diagnose HDR pipeline artifacts for a given job_id.")
	parser.add_argument("--job", required=True, help="job_id (folder name under api/normalized, api/linear, etc.)")
	args = parser.parse_args()
	run(args.job)


if __name__ == "__main__":
	main()


