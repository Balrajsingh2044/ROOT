"""TruWage backend.

Pipeline for /api/assess:
  upload -> integrity check -> AQA score + Grad-CAM heatmap -> wage range -> skill card -> passport
"""
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import schemas
from app.integrity_checker import integrity_checker
from app.nsqf_mapper import map_skills
from app.passport import build_passport, sha256_file
from app.wage_engine import demand_from_label, predict_wage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("truwage")

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
TEMP_DIR = BACKEND_DIR / "temp_uploads"
PASSPORT_DIR = BACKEND_DIR / "passports"
TEMP_DIR.mkdir(exist_ok=True)
PASSPORT_DIR.mkdir(exist_ok=True)

# The AQA model lives in indic-skill-100/. If it cannot be loaded (for example transformers is
# not installed), the server still starts and /api/assess answers with a clear 503.
sys.path.insert(0, str(REPO_ROOT / "indic-skill-100"))
try:
    from gradcam_aqa import assess_video as run_aqa
except Exception as exc:  # noqa: BLE001
    run_aqa = None
    logger.warning("AQA model unavailable: %s", exc)

ALLOWED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
MAX_UPLOAD_MB = 100

DEFAULT_ORIGINS = "http://localhost:8501,http://127.0.0.1:8501,http://localhost:7860,http://127.0.0.1:7860"
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("TRUWAGE_CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up the wage model...")
    await run_in_threadpool(predict_wage, 0.5, 3, 3)
    logger.info("AQA model available: %s", run_aqa is not None)
    yield


app = FastAPI(
    title="TruWage MVP Backend",
    description="Video integrity check, technique scoring (AQA), fair-wage range, skill card and passport",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/passports", StaticFiles(directory=str(PASSPORT_DIR)), name="passports")


def save_upload(file: UploadFile, video_id: str) -> Path:
    """Save the upload under a generated name (never the client's filename) with a size limit."""
    suffix = Path(file.filename or "").suffix.lower()
    dest = TEMP_DIR / f"{video_id}{suffix}"
    limit = MAX_UPLOAD_MB * 1024 * 1024
    size, too_big = 0, False
    with open(dest, "wb") as out:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                too_big = True
                break
            out.write(chunk)
    if too_big:  # delete after the file is closed (required on Windows)
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=f"File too large. Maximum is {MAX_UPLOAD_MB} MB.")
    return dest


@app.get("/")
def read_root():
    return {"message": "TruWage backend is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "aqa_model_loaded": run_aqa is not None}


@app.post("/get_wage", response_model=schemas.WagePredictionResponse)
async def get_wage(request: schemas.WageRequest):
    """Wage range for a given technique score, complexity and local demand."""
    wage = predict_wage(request.skill_score, request.complexity, demand_from_label(request.location_demand))
    return schemas.WagePredictionResponse(
        task_type=request.task_type,
        estimated_wage_min=wage["min"],
        estimated_wage_max=wage["max"],
        confidence_interval=f"{wage['confidence']} confidence (95% Monte-Carlo-dropout interval)",
        reasoning=(f"Skill {request.skill_score:.2f}, complexity {request.complexity}/5, "
                   f"{request.location_demand} local demand. Model trained on synthetic data."),
    )


@app.post("/api/assess", response_model=schemas.AssessResponse)
async def assess_video(
    file: UploadFile = File(...),
    task_type: str = Form("plumbing"),
    complexity: int = Form(3, ge=1, le=5),
    demand: str = Form("medium"),
    worker_name: str = Form(""),
):
    video_id = str(uuid.uuid4())
    file_path = None
    try:
        if not (file.filename or "").lower().endswith(ALLOWED_VIDEO_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
        logger.info("Received upload %s", video_id)
        file_path = save_upload(file, video_id)

        # 1. Integrity check (freeze / loop heuristic)
        integrity = await run_in_threadpool(integrity_checker.predict, str(file_path))
        if not integrity["is_authentic"]:
            logger.warning("Integrity check failed: %s", integrity["message"])
            return schemas.AssessResponse(
                deepfake_status="FAIL",
                score=integrity["confidence_score"],
                heatmap=None,
                wage_range_text="N/A",
                audio_filepath=None,
                integrity_message=integrity["message"],
            )

        # 2. Technique score + Grad-CAM heatmap
        if run_aqa is None:
            raise HTTPException(status_code=503, detail="The AQA model is not available on this server. "
                                                        "Install the requirements (transformers, torch) and restart.")
        aqa = await run_in_threadpool(run_aqa, str(file_path))
        score = min(1.0, max(0.0, float(aqa["score"])))

        # 3. Wage range from the technique score
        wage = await run_in_threadpool(predict_wage, score, complexity, demand_from_label(demand))

        # 4. Skill card
        card = map_skills("", task_type)

        # 5. Passport (optional: a failure here must not break the assessment)
        verification_hash, passport_url = None, None
        try:
            video_sha = await run_in_threadpool(sha256_file, file_path)
            passport = await run_in_threadpool(
                build_passport,
                {
                    "worker_name": worker_name.strip() or "Not provided",
                    "trade": card["trade"],
                    "nsqf_level": card["level_label"],
                    "score": f"{score:.2f}",
                    "integrity_status": "PASS",
                    "wage_range": wage["text"],
                    "wage_confidence": wage["confidence"],
                },
                video_sha, PASSPORT_DIR, video_id,
            )
            verification_hash = passport["short_hash"]
            passport_url = f"/passports/{video_id}.html"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Passport generation failed: %s", exc)

        return schemas.AssessResponse(
            deepfake_status="PASS",
            score=round(score, 4),
            heatmap=aqa.get("heatmap"),
            wage_range_text=wage["text"],
            audio_filepath=None,
            integrity_message=integrity["message"],
            trade=card["trade"],
            nsqf_level=card["level_label"],
            key_tasks=card["key_tasks"],
            wage_min=wage["min"],
            wage_max=wage["max"],
            wage_confidence=wage["confidence"],
            verification_hash=verification_hash,
            passport_url=passport_url,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Error processing video: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")
    finally:
        if file_path is not None and Path(file_path).exists():
            try:
                Path(file_path).unlink()
            except OSError as exc:
                logger.warning("Could not delete temp file: %s", exc)
