from typing import List, Optional

from pydantic import BaseModel

# --- Request Schemas (What the frontend sends) ---


class VideoUploadRequest(BaseModel):
    video_id: str
    file_name: str


class VoiceNoteRequest(BaseModel):
    audio_text: str
    language: str = "hindi"


class WageRequest(BaseModel):
    task_type: str                  # e.g. "plumbing", "tailoring"
    complexity: int                 # 1 to 5
    location_demand: str            # "low", "medium" or "high"
    skill_score: float = 0.5        # technique score 0-1 (the video pipeline fills this in)
    voice_note: Optional[VoiceNoteRequest] = None


# --- Response Schemas (What the backend sends back) ---


class IntegrityCheckResponse(BaseModel):
    video_id: str
    is_authentic: bool
    confidence_score: float
    message: str


class WagePredictionResponse(BaseModel):
    task_type: str
    estimated_wage_min: float
    estimated_wage_max: float
    confidence_interval: str
    reasoning: str


class FullPipelineResponse(BaseModel):
    integrity_check: IntegrityCheckResponse
    wage_prediction: Optional[WagePredictionResponse] = None
    overall_status: str


class AssessResponse(BaseModel):
    """The original five fields (the Gradio dashboard needs them) plus the new optional fields."""
    deepfake_status: str                 # "PASS" or "FAIL" (name kept for the dashboard)
    score: float                         # technique score 0-1 (integrity confidence if FAIL)
    heatmap: Optional[str] = None        # data-URI image
    wage_range_text: str                 # e.g. "240 - 270 rupees per hour" or "N/A"
    audio_filepath: Optional[str] = None
    # --- new ---
    integrity_message: Optional[str] = None
    trade: Optional[str] = None
    nsqf_level: Optional[str] = None
    key_tasks: Optional[List[str]] = None
    wage_min: Optional[int] = None
    wage_max: Optional[int] = None
    wage_confidence: Optional[str] = None
    verification_hash: Optional[str] = None
    passport_url: Optional[str] = None
