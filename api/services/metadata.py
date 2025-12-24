from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
import piexif  # relies on requirements.txt

from api.services.image_utils import apply_exif_orientation


def _rational_to_float(x: Any) -> Optional[float]:
	if x is None:
		return None
	if isinstance(x, tuple) and len(x) == 2:
		num, den = x
		if not den:
			return None
		return float(num) / float(den)
	try:
		return float(x)
	except Exception:
		return None


def _apex_to_time(apex: Optional[float]) -> Optional[float]:
	return 2.0 ** (-apex) if apex is not None else None


def _mean_brightness(img: Image.Image) -> float:
	g = img.convert("L")
	data = g.getdata()
	return sum(data) / (g.width * g.height)


def _bytes_to_str(v: Any) -> Optional[str]:
	if v is None:
		return None
	if isinstance(v, bytes):
		try:
			return v.decode("utf-8", errors="ignore")
		except Exception:
			return None
	if isinstance(v, str):
		return v
	return str(v)


def _to_int_safe(v: Any) -> Optional[int]:
	if v is None:
		return None
	if isinstance(v, (int,)):
		return v
	if isinstance(v, (list, tuple)) and v:
		try:
			return int(v[0])
		except Exception:
			return None
	if isinstance(v, bytes):
		try:
			s = v.decode("utf-8", errors="ignore").strip()
			return int(s) if s else None
		except Exception:
			return None
	try:
		return int(v)
	except Exception:
		return None


def extract_metadata(saved_paths: List[Path]) -> Dict[str, Any]:
	records: List[Dict[str, Any]] = []
	for p in saved_paths:
		img = Image.open(p)
		img = apply_exif_orientation(img, img.getexif())
		# base info
		info: Dict[str, Any] = {
			"filename": p.name,
			"width": img.width,
			"height": img.height,
			"mode": img.mode,
			"mean_brightness": _mean_brightness(img),
		}
		# EXIF via piexif
		try:
			ex = piexif.load(str(p))
			exif = ex.get("Exif", {})
			# exposure time
			exp = _rational_to_float(exif.get(piexif.ExifIFD.ExposureTime))
			if exp is None:
				ss_apex = _rational_to_float(exif.get(piexif.ExifIFD.ShutterSpeedValue))
				exp = _apex_to_time(ss_apex)
			info["exposure_time_s"] = exp
			# f-number
			info["fnumber"] = _rational_to_float(exif.get(piexif.ExifIFD.FNumber))
			# ISO
			iso = exif.get(piexif.ExifIFD.ISOSpeedRatings)
			info["iso"] = _to_int_safe(iso)
			# orientation (from 0th IFD)
			zeroth = ex.get("0th", {})
			info["orientation"] = zeroth.get(piexif.ImageIFD.Orientation)
			# datetime
			dt = exif.get(piexif.ExifIFD.DateTimeOriginal)
			if not dt:
				dt = zeroth.get(piexif.ImageIFD.DateTime)
			info["datetime_original"] = _bytes_to_str(dt)
		except Exception:
			# If EXIF missing or unreadable, keep base info only
			pass
		records.append(info)
	return {"images": records}


def write_metadata_json(metadata: Dict[str, Any], out_path: Path) -> str:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2)
	return str(out_path)

