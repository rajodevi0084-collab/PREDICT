"""Data ingestion and inspection endpoints."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.ml.data_loader import DATA_DIR, SUPPORTED_EXTENSIONS, load_dataset

router = APIRouter(prefix="/data", tags=["data"])


def _resolve_file(file_id: str) -> Path:
    for suffix in sorted(SUPPORTED_EXTENSIONS):
        candidate = DATA_DIR / f"{file_id}{suffix}"
        if candidate.exists():
            return candidate
    matches = list(DATA_DIR.glob(f"{file_id}.*"))
    if matches:
        return matches[0]
    raise HTTPException(status_code=404, detail="Dataset not found")


@router.post("/upload")
async def upload_data(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a name")

    original_suffix = Path(file.filename).suffix.lower()
    suffix = original_suffix if original_suffix in SUPPORTED_EXTENSIONS else ""

    if not suffix:
        content_type = (file.content_type or "").lower()
        if "csv" in content_type:
            suffix = ".csv"
        elif "parquet" in content_type:
            suffix = ".parquet"

    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only CSV and Parquet uploads are supported")

    file_id = uuid.uuid4().hex
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / f"{file_id}{suffix}"

    file.file.seek(0)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df, metadata = load_dataset(destination)
    except Exception as exc:  # pragma: no cover - defensive cleanup
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset: {exc}") from exc

    preview_rows = df.head(50).to_dict(orient="records")

    return {
        "file_id": file_id,
        "filename": file.filename,
        "stored_as": destination.name,
        "metadata": metadata.to_dict(),
        "preview": preview_rows,
    }


@router.get("/preview/{file_id}")
async def preview_dataset(file_id: str) -> dict[str, object]:
    path = _resolve_file(file_id)
    df, metadata = load_dataset(path)
    rows = df.head(50).to_dict(orient="records")
    return {
        "file_id": file_id,
        "filename": path.name,
        "rows": rows,
        "dtypes": metadata.dtypes,
    }


@router.get("/catalog")
async def catalog() -> dict[str, object]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, object]] = []
    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        _, metadata = load_dataset(path)
        items.append(
            {
                "file_id": path.stem,
                "filename": path.name,
                "metadata": metadata.to_dict(),
            }
        )
    return {"items": items}
