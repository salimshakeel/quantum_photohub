from __future__ import annotations

from datetime import datetime
from pathlib import Path
import uuid
from io import BytesIO
from typing import List

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile

from api.services.upload_pipeline import run_pipeline
from api.services.status_store import read_status, write_status


router = APIRouter(prefix="/pipeline", tags=["upload"])


def _slugify(text: str) -> str:
	return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "-" for ch in text).strip("-_").lower()


@router.post("/upload", summary="Upload bracketed images and start background processing")
async def upload(
	background_tasks: BackgroundTasks,
	files: List[UploadFile] = File(...),
	scene: str = Form("bracket_001/uploaded"),
):
	files_meta = []
	for f in files:
		data = await f.read()
		files_meta.append({"filename": f.filename or "image.jpg", "data": data})
	filenames = [m["filename"] for m in files_meta]
	# Human-readable job_id: "<first_filename_stem>_<ddmmyyyy>"
	first_stem = _slugify(Path(filenames[0]).stem) if filenames else "job"
	date_str = datetime.now().strftime("%d%m%Y")
	job_id = f"{first_stem}_{date_str}"
	write_status(job_id, {"job_id": job_id, "status": "queued", "step": "Queued"})
	# Always process in linear-light for correct HDR math
	linearize = True
	background_tasks.add_task(run_pipeline, job_id, scene, files_meta, linearize)
	return {
		"job_id": job_id,
		"status": "queued",
		"scene": scene,
		"num_files": len(files_meta),
		"filenames": filenames,
		"status_endpoint": f"/pipeline/status/{job_id}",
		"result_endpoint": f"/pipeline/result/{job_id}",
	}


@router.get("/status/{job_id}", summary="Get pipeline status")
def status(job_id: str):
	return read_status(job_id)


@router.get("/result/{job_id}", summary="Get pipeline results")
def result(job_id: str):
	data = read_status(job_id)
	if data.get("status") != "completed":
		return {"job_id": job_id, "status": data.get("status"), "message": "not completed yet"}
	return {
		"job_id": job_id,
		"metadata": data.get("metadata"),
		"proposed_order": data.get("proposed_order", []),
		"validation": data.get("validation", {}),
		"normalized": data.get("normalized", []),
		"linear_dir": data.get("linear_dir"),
		"aligned": data.get("aligned", []),
		
	}

