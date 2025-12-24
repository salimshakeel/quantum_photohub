from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
from api.services.status_store import write_status
from api.services.normalization import normalize_set
from api.services.metadata import extract_metadata, write_metadata_json
from api.services.alignment import align_set
from api.services.fusion import run_exposure_fusion_from_aligned


def run_pipeline(job_id: str, scene: str, files_meta: List[Dict[str, Any]], linearize: bool) -> None:
	try:
		# 1) Save originals to api/input/<job_id>/
		write_status(job_id, {"job_id": job_id, "status": "saving", "step": "Save Images"})
		in_dir = Path("api/input") / job_id
		in_dir.mkdir(parents=True, exist_ok=True)
		saved = []
		for fm in files_meta:
			name = Path(fm["filename"]).name
			p = in_dir / name
			with p.open("wb") as f:
				f.write(fm["data"])
			saved.append(p)

		# 2) Extract metadata and write JSON
		write_status(job_id, {"job_id": job_id, "status": "metadata", "step": "Extract Metadata"})
		metadata = extract_metadata(saved)
		metadata_path = write_metadata_json(metadata, Path("api/input") / job_id / "metadata.json")

		# 2.1) Validate inputs and derive proposed order
		records = metadata.get("images", [])
		# size validation (after orientation fix applied during metadata extraction)
		size_set = {(int(r.get("width", 0)), int(r.get("height", 0))) for r in records}
		same_resolution = len(size_set) == 1
		# orientation info
		orientations = [r.get("orientation") for r in records]
		orientation_values = {o for o in orientations if o is not None}
		orientation_fixed = any(o not in (1, None) for o in orientations)
		# exposure ordering (fallback to mean_brightness if exposure_time_s missing)
		exp_pairs = []
		for r, p in zip(records, saved):
			et = r.get("exposure_time_s")
			mb = r.get("mean_brightness")
			exp_pairs.append((et, mb, p))
		if any(et is not None for (et, _, _) in exp_pairs):
			sorted_pairs = sorted(exp_pairs, key=lambda t: (float("inf") if t[0] is None else t[0]))
		else:
			sorted_pairs = sorted(exp_pairs, key=lambda t: (float("inf") if t[1] is None else t[1]))
		sorted_saved = [p for _, _, p in sorted_pairs]
		proposed_order = [p.name for p in sorted_saved]

		validation = {
			"same_resolution": same_resolution,
			"unique_resolutions": list(sorted({"{}x{}".format(w, h) for (w, h) in size_set})),
			"exif_orientations_present": sorted(list(orientation_values)),
			"orientation_fixed_from_exif": orientation_fixed,
		}

		write_status(job_id, {
			"job_id": job_id,
			"status": "validated",
			"step": "Validate Inputs",
			"metadata": metadata_path,
			"proposed_order": proposed_order,
			"validation": validation,
		})

		# 3) Normalize images (size/RGB/float) using proposed order
		write_status(job_id, {
			"job_id": job_id,
			"status": "normalizing",
			"step": "Normalize Images",
			"metadata": metadata_path,
		})
		# Save normalized outputs:
		# - PNGs to api/normalized/<job_id>/
		# - Linear float arrays to api/linear/<job_id>/
		norm_dir = Path("api/normalized") / job_id
		linear_dir = Path("api/linear") / job_id
		norm_out = normalize_set(sorted_saved, norm_dir, linear_dir, linearize=linearize)

		# 4) Align normalized linear arrays using MTB (translation)
		write_status(job_id, {
			"job_id": job_id,
			"status": "aligning",
			"step": "Align Images (MTB)",
			"metadata": metadata_path,
			"proposed_order": proposed_order,
			"validation": validation,
			"normalized": norm_out,
			"linear_dir": str(linear_dir),
		})
		npy_paths = [linear_dir / f"{p.stem}_linear.npy" for p in sorted_saved]
		ref_index = len(npy_paths) // 2
		align_dir = Path("api/aligned") / job_id
		align_res = align_set(job_id, npy_paths, ref_index, align_dir, save_png=False)
		aligned_paths: List[str] = list(align_res.get("aligned_paths", []))  # type: ignore[assignment]
		transforms_path: str = str(align_res.get("transforms", ""))

		# 5) Exposure Fusion (LDR)
		write_status(job_id, {
			"job_id": job_id,
			"status": "fusing",
			"step": "Exposure Fusion",
			"metadata": metadata_path,
			"proposed_order": proposed_order,
			"validation": validation,
			"normalized": norm_out,
			"linear_dir": str(linear_dir),
			"aligned": aligned_paths,
			"transforms": transforms_path,
		})
		fused_dir = Path("api/fused") / job_id
		fused_path = run_exposure_fusion_from_aligned([Path(p) for p in aligned_paths], fused_dir / "fused.png")

		# 6) Complete
		write_status(job_id, {
			"job_id": job_id,
			"status": "completed",
			"step": "Done",
			"metadata": metadata_path,
			"proposed_order": proposed_order,
			"validation": validation,
			"normalized": norm_out,
			"linear_dir": str(linear_dir),
			"aligned": aligned_paths,
			"transforms": transforms_path,
			"fused": fused_path,
		})
	except Exception as e:
		write_status(job_id, {"job_id": job_id, "status": "error", "error": str(e)})

