from fastapi import FastAPI, UploadFile, File, HTTPException
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
import uuid


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

# --- 1. Check Integrity Endpoint (Deepfake Detection) ---
@app.post("/check_integrity", response_model=schemas.IntegrityCheckResponse)
async def check_integrity(video_id: str, file_path: str):
    """
    Day 2 Update: Uses the pre-trained 3D CNN + heuristic to check video integrity.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Run the AI check
    result = integrity_checker.predict(file_path)
    
    return schemas.IntegrityCheckResponse(
        video_id=video_id,
        is_authentic=result["is_authentic"],
        confidence_score=result["confidence_score"],
        message=result["message"]
    )  

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
    task_type: str = "plumbing",
    complexity: int = 3
):
    """
    Main pipeline: Saves video, checks integrity, short-circuits if fake, else predicts wage.
    """
    video_id = str(uuid.uuid4())
    
    # 1. Save the uploaded file temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{video_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Check Integrity using our new Day 2 model
    integrity_result = await check_integrity(video_id=video_id, file_path=file_path)
    
    # 3. Short-circuit logic: If fake, DO NOT run the heavy AQA/Wage models
    if not integrity_result.is_authentic:
        # Clean up the file
        os.remove(file_path)
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
    
    # Clean up the file after processing
    os.remove(file_path)
    
    return schemas.FullPipelineResponse(
        integrity_check=integrity_result,
        wage_prediction=wage_result,
        overall_status="SUCCESS"
    )

# --- Health Check (Good practice for debugging) ---
@app.get("/")
def read_root():
    return {"message": "TruWage Backend is running!"}

@app.post("/api/assess", response_model=schemas.AssessResponse)
async def assess_video(
    file: UploadFile = File(...),
    task_type: str = "plumbing",
    complexity: int = 3
):
    """
    The Master Endpoint for Balraj's Frontend.
    """
    video_id = str(uuid.uuid4())
    file_path = None
    
    try:
        logger.info(f"Received video upload: {file.filename}")
        
        # 1. Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
        
        # 2. Save the uploaded file temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, f"{video_id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Video saved to {file_path}")
        
        # 3. Run the Deepfake Check
        integrity_result = integrity_checker.predict(file_path)
        
        # 4. Format the response
        if not integrity_result["is_authentic"]:
            logger.warning(f"Video failed integrity check: {integrity_result['message']}")
            return schemas.AssessResponse(
                deepfake_status="FAIL",
                score=integrity_result["confidence_score"],
                heatmap=None,
                wage_range_text="N/A",
                audio_filepath=None
            )
        
        # 5. Calculate mock wage
        base_wage = 500
        min_wage = base_wage * (complexity * 0.5)
        max_wage = min_wage * 1.2
        
        logger.info(f"Video passed integrity check. Score: {integrity_result['confidence_score']}")
        
        return schemas.AssessResponse(
            deepfake_status="PASS",
            score=integrity_result["confidence_score"],
            heatmap="https://via.placeholder.com/400x300.png?text=Grad-CAM+Heatmap+Coming+Soon",
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