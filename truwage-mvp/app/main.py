"""TruWage backend.

Pipeline for /api/assess:
  upload -> integrity check -> AQA score + Grad-CAM heatmap -> wage range -> skill card -> passport
"""
import asyncio
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import schemas
from app.integrity_checker import integrity_checker
from app.nsqf_mapper import map_skills
from app.passport import build_passport, sha256_file
from app.transcriber import transcribe_audio
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
ALLOWED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac")
MAX_UPLOAD_MB = 100

DEFAULT_ORIGINS = "http://localhost:8501,http://127.0.0.1:8501,http://localhost:7860,http://127.0.0.1:7860"
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("TRUWAGE_CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    sweep_temp_files()
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


async def _remove_file(path, attempts: int = 8, delay: float = 0.25) -> bool:
    """Delete a temp file, retrying briefly: Windows can lock a just-written file (antivirus, indexer)."""
    for _ in range(attempts):
        try:
            Path(path).unlink(missing_ok=True)
            return True
        except OSError:
            await asyncio.sleep(delay)
    logger.warning("Could not delete temp file %s (it will be swept later)", path)
    return False


def sweep_temp_files(max_age_seconds: int = 3600) -> None:
    """Remove stale uploads left behind by crashes or locked files."""
    now = time.time()
    for p in TEMP_DIR.glob("*"):
        try:
            if p.is_file() and now - p.stat().st_mtime > max_age_seconds:
                p.unlink()
        except OSError:
            pass


def save_upload(file: UploadFile, stem: str) -> Path:
    """Save the upload under a generated name (never the client's filename) with a size limit."""
    suffix = Path(file.filename or "").suffix.lower()
    dest = TEMP_DIR / f"{stem}{suffix}"
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
    audio: Optional[UploadFile] = File(None),
    language: str = Form("hindi"),
):
    video_id = str(uuid.uuid4())
    file_path = None
    audio_path = None
    try:
        if not (file.filename or "").lower().endswith(ALLOWED_VIDEO_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
        logger.info("Received upload %s", video_id)
        file_path = save_upload(file, video_id)

        has_audio = audio is not None and bool(audio.filename)
        if has_audio:
            if not audio.filename.lower().endswith(ALLOWED_AUDIO_EXTENSIONS):
                raise HTTPException(status_code=400, detail="Invalid audio type. Use wav, mp3, m4a, ogg, flac or aac.")
            audio_path = save_upload(audio, f"{video_id}_audio")

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

        # 4. Voice note transcript (optional; a failure here must not break the assessment)
        transcript, transcript_note, voice_trade = None, None, None
        if audio_path is not None:
            try:
                transcript = await run_in_threadpool(transcribe_audio, str(audio_path), language)
                if not transcript:
                    transcript_note = "No speech was recognised in the audio."
            except Exception as exc:  # noqa: BLE001
                logger.warning("Transcription failed: %s", exc)
                transcript_note = f"Transcription failed: {exc}"

        # 5. Skill card (the selected trade decides; the transcript adds the matched words)
        card = map_skills(transcript or "", task_type)
        if transcript:
            heard = map_skills(transcript, "")
            if heard["source"] == "transcript":
                voice_trade = heard["trade"]
                if heard["trade_key"] != card["trade_key"]:
                    transcript_note = (f"The voice note sounds like {heard['trade']}, "
                                       f"but the selected trade is {card['trade']}.")

        # 6. Passport (optional: a failure here must not break the assessment)
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
            transcript=transcript,
            transcript_note=transcript_note,
            voice_trade=voice_trade,
            matched_keywords=card["matched_keywords"],
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Error processing video: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")
    finally:
        for tmp in (file_path, audio_path):
            if tmp is not None:
                await _remove_file(tmp)
