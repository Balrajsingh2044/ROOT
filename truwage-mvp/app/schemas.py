from pydantic import BaseModel
from typing import Optional, List

# --- Request Schemas (What the frontend sends) ---

class VideoUploadRequest(BaseModel):
    video_id: str
    file_name: str
    # In a real app, the file itself is sent as multipart/form-data, 
    # but this schema validates the metadata.

class VoiceNoteRequest(BaseModel):
    audio_text: str
    language: str = "hindi" # Default to hindi for the mock

class WageRequest(BaseModel):
    task_type: str          # e.g., "plumbing", "tailoring"
    complexity: int         # 1 to 5
    location_demand: str    # e.g., "high", "medium", "low"
    voice_note: Optional[VoiceNoteRequest] = None

# --- Response Schemas (What the backend sends back) ---

class IntegrityCheckResponse(BaseModel):
    video_id: str
    is_authentic: bool      # True = Real, False = Fake/Looped
    confidence_score: float # e.g., 0.95
    message: str

class WagePredictionResponse(BaseModel):
    task_type: str
    estimated_wage_min: float
    estimated_wage_max: float
    confidence_interval: str # e.g., "85% confidence"
    reasoning: str

class FullPipelineResponse(BaseModel):
    integrity_check: IntegrityCheckResponse
    wage_prediction: Optional[WagePredictionResponse] = None
    overall_status: str # "SUCCESS" or "FAILED_INTEGRITY_CHECK"


class AssessResponse(BaseModel):
    """
    The exact 5 variables Balraj's dashboard.py expects.
    """
    deepfake_status: str      # e.g., "PASS" or "FAIL"
    score: float              # e.g., 0.92
    heatmap: str              # URL or base64 string of the heatmap image (Ayansh will provide this later)
    wage_range_text: str      # e.g., "₹500 - ₹600"
    audio_filepath: str       # URL or local path to the TTS audio file (Balraj will provide this later)