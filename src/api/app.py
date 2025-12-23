from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
import uvicorn
import asyncio
import glob
import aiofiles

from src.core.config import settings, SystemMode
from src.infra.logging import get_logger
from src.core.engine import InspectionEngine
from src.core.dataset import DatasetManager
from src.hal.camera.source import PiCameraSource
from src.hal.files.source import FileImageSource
from fastapi import File, UploadFile
from fastapi.responses import FileResponse
from src.core.types import InspectionResult, InspectionResultType
from src.infra.storage import ImageStore
from src.infra.database import SessionLocal, InspectionDB

logger = get_logger(__name__)

app = FastAPI(title="LCD Inspection Node", version="1.0.0")

# Global instances
engine: Optional[InspectionEngine] = None
dataset_manager: Optional[DatasetManager] = None
image_source = None

# Models
class OverrideRequest(BaseModel):
    original_result: str
    new_result: str
    reason: Optional[str] = None

class CommitRequest(BaseModel):
    version_tag: str

@app.on_event("startup")
async def startup_event():
    global engine, dataset_manager, image_source
    logger.info("API Startup...")

    dataset_manager = DatasetManager()
    engine = InspectionEngine()

    # Load Reference Image for Vision Engine
    # In a real app, this logic would be in a shared Controller or Service.
    # For now, we bridge Dataset -> Engine here.
    if dataset_manager.active_version_id:
        # We need to find the reference image path for this dataset.
        # Assuming the dataset stores config_json with "reference_image_path"
        from src.infra.database import SessionLocal, DatasetDB
        with SessionLocal() as db:
            ds = db.query(DatasetDB).filter(DatasetDB.id == dataset_manager.active_version_id).first()
            if ds and ds.config_json and "reference_image_path" in ds.config_json:
                ref_path = settings.DATA_DIR / ds.config_json["reference_image_path"]
                if ref_path.exists():
                    engine.load_reference(str(ref_path))
                else:
                    logger.error(f"Reference image not found at {ref_path}")
            else:
                 logger.warning("Active dataset has no reference image defined.")

    # Initialize Source based on Mode
    if settings.MODE == SystemMode.PRODUCTION:
        image_source = PiCameraSource()
    else:
        # Default dev folder - ensure distinct from output store
        dev_input_path = settings.DATA_DIR / "input_sim"
        dev_input_path.mkdir(parents=True, exist_ok=True)
        image_source = FileImageSource(dev_input_path)

    image_source.initialize()
    logger.info(f"Initialized Image Source: {type(image_source).__name__}")

@app.on_event("shutdown")
async def shutdown_event():
    if image_source:
        image_source.release()

@app.get("/status")
def get_status():
    return {
        "status": "online",
        "mode": settings.MODE,
        "camera_ready": image_source.camera_ready if hasattr(image_source, "camera_ready") else True
    }

@app.post("/inspect", response_model=InspectionResult)
async def trigger_inspection(background_tasks: BackgroundTasks):
    """
    Triggers an inspection.
    In a real async design, we might offload this to a worker,
    but for <2.5s latency, direct execution is okay if we don't block the event loop too hard.
    However, subprocess calls and CV are blocking.
    We should use run_in_executor for the heavy lifting.
    """
    loop = asyncio.get_event_loop()

    try:
        # 1. Capture
        # Run in executor to avoid blocking async loop
        raw_image = await loop.run_in_executor(None, image_source.capture)

        # 2. Process
        result = await loop.run_in_executor(None, engine.process, raw_image)

        # 3. Save Image (Background Task)
        background_tasks.add_task(ImageStore.save_image, raw_image)

        # 4. Save Result to DB (Sync for now, could be async/background)
        # We do this here to ensure API response includes confirmation if needed,
        # but technically logging can be fire-and-forget.
        # Given "Traceability", we should ensure it's written.
        save_result_to_db(result, raw_image)

        return result

    except Exception as e:
        logger.error(f"Inspection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/results/{id}/override")
def override_result(id: UUID, req: OverrideRequest):
    try:
        dataset_manager.add_override(id, req.original_result, req.new_result, req.reason)
        return {"status": "pending", "id": id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dataset/commit")
def commit_dataset(req: CommitRequest):
    try:
        dataset_manager.commit_version(req.version_tag)
        return {"status": "committed", "version": req.version_tag}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/last", response_model=Optional[InspectionResult])
def get_last_result():
    try:
        with SessionLocal() as db:
            last = db.query(InspectionDB).order_by(InspectionDB.timestamp.desc()).first()
            if not last:
                return None
            return InspectionResult(
                id=UUID(last.id),
                image_id=UUID(last.image_path) if last.image_path else UUID('00000000-0000-0000-0000-000000000000'), # Handling generic path as ID
                timestamp=last.timestamp,
                result=InspectionResultType(last.result),
                confidence_score=last.score,
                processing_time_ms=0, # Not stored in main columns, arguably in meta
                dataset_version=last.dataset_version,
                details=last.meta_json or {}
            )
    except Exception as e:
        logger.error(f"Error fetching last result: {e}")
        raise HTTPException(status_code=500, detail="Internal Error")

@app.get("/images/{id}")
def get_image(id: UUID):
    # Locate image. Since we store by date, we might need to query DB to find path if not strictly determinstic by ID alone without date.
    # Current storage: YYYY/MM/DD/ID.jpg.
    # We can glob search or assume we look up in DB.
    # For MVP, we'll try to find it via DB lookup or search.
    # DB has image_path which we stored as ID in `save_result_to_db`.
    # Wait, in `save_result_to_db` we set `image_path=str(raw_image.id)`.
    # `storage.py` returns `relative_path` (including date). We should have stored THAT.
    # Fix `save_result_to_db` to store actual path if possible.
    # But `save_image` is background.

    # Fallback search strategy:
    # Look in data/images/**/*.jpg with that name.

    pattern = str(settings.IMAGE_STORE_PATH / "**" / f"{id}.jpg")
    files = list(glob.glob(pattern, recursive=True))
    if not files:
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(files[0])

@app.post("/upload")
async def upload_test_image(file: UploadFile = File(...)):
    if settings.MODE != SystemMode.DEVELOPER:
        raise HTTPException(status_code=400, detail="Upload only allowed in Developer Mode")

    try:
        # Save to dev input folder
        dev_path = settings.DATA_DIR / "input_sim"
        dev_path.mkdir(parents=True, exist_ok=True)
        file_path = dev_path / file.filename

        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        return {"status": "uploaded", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def save_result_to_db(result: InspectionResult, raw_image: ImageStore):
    try:
        with SessionLocal() as db:
            db_record = InspectionDB(
                id=str(result.id),
                timestamp=result.timestamp,
                image_path=str(raw_image.id), # We might want the path, but ID is stable. Storage manages path.
                result=result.result.value,
                score=result.confidence_score,
                dataset_version=result.dataset_version,
                meta_json=result.details
            )
            # Update image path if we have the saved path, but saving is async background...
            # This is a race condition if we want the file path in DB immediately.
            # Strategy: Use deterministic path or update later.
            # Ideally, `save_image` returns the path, but it's in a background task.
            # Fix: Save image synchronously here OR rely on ID.
            # Given latency constraint (<2.5s), saving 64MP image (5-10MB) takes time.
            # We will store the ID. The `storage.py` logic maps ID -> Path deterministically.

            db.add(db_record)
            db.commit()
    except Exception as e:
        logger.error(f"Failed to save inspection result to DB: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
