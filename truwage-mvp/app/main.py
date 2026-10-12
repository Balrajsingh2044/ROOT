from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool  # CHANGED: run heavy model calls in a thread
from app import schemas
from app.deepfake_detector import integrity_checker
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import shutil
import uuid
import os
from fastapi.staticfiles import StaticFiles
import sys
from fastapi.concurrency import run_in_threadpool

AQA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../indic-skill-100"))
sys.path.insert(0, AQA_DIR)
from gradcam_aqa import assess_video as run_aqa

# CHANGED: added .webm so webcam recordings are accepted
ALLOWED_VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event: Runs once on startup and once on shutdown.
    """
    # Startup: Load models
    logger.info("Starting up: Loading AI models...")
    # The integrity_checker is already loaded at module level, but we can add more models here later
    logger.info("All models loaded successfully!")
    
    yield  # This is where the app runs
    
    # Shutdown: Clean up
    logger.info("Shutting down: Cleaning up resources...")
    # If we had database connections or other resources, we'd close them here

# Initialize the FastAPI app
app = FastAPI(
    title="TruWage MVP Backend",
    description="API for Video Integrity, AQA, and Fair Wage Prediction",
    version="1.0.0",
    lifespan = lifespan 
)

# Allow Balraj's frontend to talk to your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this. For MVP hackathon, "*" is fine.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a folder to hold heatmaps at the ROOT level
os.makedirs(os.path.join(os.path.dirname(__file__), "../../heatmaps"), exist_ok=True)

# Tell FastAPI to serve files from the ROOT/heatmaps folder
app.mount("/heatmaps", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../../heatmaps")), name="heatmaps")


# CHANGED: helper that saves an uploaded video to a temp file (replaces duplicated save code)
def save_upload(file: UploadFile, video_id: str) -> str:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{video_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path


# --- 1. Check Integrity Endpoint (Deepfake Detection) ---
# CHANGED: the public /check_integrity route was removed because it accepted an arbitrary
# file_path from the client (arbitrary file read risk). The integrity check now runs
# directly via integrity_checker.predict() inside the endpoints below.

# --- 2. Get Wage Endpoint (GNN Fair-Wage Engine) ---
@app.post("/get_wage", response_model=schemas.WagePredictionResponse)
async def get_wage(request: schemas.WageRequest):
    """
    Dummy endpoint for Day 1.
    Day 2/3 Goal: Replace with Bayesian GNN prediction logic.
    """
    # MOCK LOGIC: Simple hardcoded logic based on complexity
    base_wage = 500
    multiplier = request.complexity * 1.5
    
    min_wage = base_wage * multiplier
    max_wage = min_wage * 1.2 # 20% upper bound for Bayesian range
    
    return schemas.WagePredictionResponse(
        task_type=request.task_type,
        estimated_wage_min=min_wage,
        estimated_wage_max=max_wage,
        confidence_interval="85% confidence (Monte Carlo Dropout)",
        reasoning=f"Wage adjusted for {request.complexity}/5 complexity and {request.location_demand} local demand."
    )
# -- Health ---
@app.get("/health")
def health_check():
    """
    Simple health check endpoint. Balraj's frontend can ping this to verify backend is alive.
    """
    return {
        "status": "healthy",
        "message": "TruWage backend is running",
        "models_loaded": True
    }
# --- 3. Full Pipeline Endpoint (Upload Video + Voice) ---
@app.post("/upload_video", response_model=schemas.FullPipelineResponse)
async def upload_video(
    file: UploadFile = File(...),
    task_type: str = Form("plumbing"),   # CHANGED: Form so it's read from the form data
    complexity: int = Form(3)            # CHANGED: Form so it's read from the form data
):
    """
    Main pipeline: Saves video, checks integrity, short-circuits if fake, else predicts wage.
    """
    video_id = str(uuid.uuid4())
    
    # 1. Save the uploaded file temporarily
    file_path = save_upload(file, video_id)
    
    try:
        # 2. Check Integrity using our new Day 2 model
        # CHANGED: runs in a thread, and calls the model directly (no /check_integrity route)
        result = await run_in_threadpool(integrity_checker.predict, file_path)
        integrity_result = schemas.IntegrityCheckResponse(
            video_id=video_id,
            is_authentic=result["is_authentic"],
            confidence_score=result["confidence_score"],
            message=result["message"]
        )
        
        # 3. Short-circuit logic: If fake, DO NOT run the heavy AQA/Wage models
        if not integrity_result.is_authentic:
            return schemas.FullPipelineResponse(
                integrity_check=integrity_result,
                overall_status="FAILED_INTEGRITY_CHECK"
            )
        
        # 4. If authentic, get wage prediction (Mock for now, Balraj will plug in GNN later)
        wage_request = schemas.WageRequest(
            task_type=task_type,
            complexity=complexity,
            location_demand="high"
        )
        wage_result = await get_wage(wage_request)
        
        return schemas.FullPipelineResponse(
            integrity_check=integrity_result,
            wage_prediction=wage_result,
            overall_status="SUCCESS"
        )
    finally:
        # Clean up the file after processing (CHANGED: now also runs on fake videos and errors)
        if os.path.exists(file_path):
            os.remove(file_path)

# --- Health Check (Good practice for debugging) ---
@app.get("/")
def read_root():
    return {"message": "TruWage Backend is running!"}

@app.post("/api/assess", response_model=schemas.AssessResponse)
async def assess_video(
    file: UploadFile = File(...),
    task_type: str = Form("plumbing"),   # CHANGED: Form so the dashboard's values are actually received
    complexity: int = Form(3)            # CHANGED: Form so the dashboard's values are actually received
):
    """
    The Master Endpoint for Frontend.
    """
    video_id = str(uuid.uuid4())
    file_path = None
    
    try:
        logger.info(f"Received video upload: {file.filename}")
        
        # 1. Validate file type
        if not file.filename.lower().endswith(ALLOWED_VIDEO_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
        
        # 2. Save the uploaded file temporarily
        file_path = save_upload(file, video_id)
        
        logger.info(f"Video saved to {file_path}")
        
        # 3. Run the Deepfake Check (Your 3D CNN)
        # CHANGED: runs in a thread so the server isn't blocked during inference
        integrity_result = await run_in_threadpool(integrity_checker.predict, file_path)
        
        # 4. Format the response if FAKE
        if not integrity_result["is_authentic"]:
            logger.warning(f"Video failed integrity check: {integrity_result['message']}")
            # set to FAIL, and don't calculate wage/heatmap
            return schemas.AssessResponse(
                deepfake_status="FAIL",
                score=integrity_result["confidence_score"],
                heatmap=None,  
                wage_range_text="N/A",
                audio_filepath=None 
            )
        
        # 5. Real AQA score + Grad-CAM heatmap computed from THIS video
        aqa = await run_in_threadpool(run_aqa, file_path)
        aqa_score = aqa["score"]      # 0–1
        heatmap = aqa["heatmap"]      # base64 data-URI (jpeg)
        
        # 6. Calculate mock wage
        base_wage = 500
        min_wage = base_wage * (complexity * 0.5)
        max_wage = min_wage * 1.2
        
        logger.info(f"Video passed integrity check. AQA Score: {aqa_score}")
        
        return schemas.AssessResponse(
            deepfake_status="PASS",
            score=aqa_score,
            heatmap=heatmap,
            wage_range_text=f"₹{int(min_wage)} - ₹{int(max_wage)}",
            audio_filepath="https://www.soundjay.com/buttons/sounds/button-09.mp3"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Always clean up the file, even if there's an error
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")