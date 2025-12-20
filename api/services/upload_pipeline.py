from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
from api.services.status_store import write_status
from api.services.previews import generate_previews
from api.services.normalization import normalize_set
from api.services.metadata import extract_metadata, write_metadata_json


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

		# 3) Normalize images (size/RGB/float)
		write_status(job_id, {
			"job_id": job_id,
			"status": "normalizing",
			"step": "Normalize Images",
			"metadata": metadata_path,
		})
		norm_out = normalize_set(saved, Path("hdr_pipeline/preprocessed") / job_id / "normalized", linearize=linearize)

		# 4) Complete
		write_status(job_id, {
			"job_id": job_id,
			"status": "completed",
			"step": "Done",
			"metadata": metadata_path,
			"normalized": norm_out,
		})
	except Exception as e:
		write_status(job_id, {"job_id": job_id, "status": "error", "error": str(e)})

