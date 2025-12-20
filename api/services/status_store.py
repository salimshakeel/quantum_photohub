from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def write_status(job_id: str, data: Dict[str, Any]) -> None:
	status_path = JOBS_DIR / f"{job_id}.json"
	with status_path.open("w", encoding="utf-8") as f:
		json.dump(data, f, indent=2)


def read_status(job_id: str) -> Dict[str, Any]:
	status_path = JOBS_DIR / f"{job_id}.json"
	if not status_path.exists():
		return {"job_id": job_id, "status": "unknown"}
	with status_path.open("r", encoding="utf-8") as f:
		return json.load(f)

