import sys
from pathlib import Path

# --- 1. Path Setup for AQA ---
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent # Points to your main 'indic' folder
sys.path.append(str(ROOT_DIR / "indic-skill-100"))

# --- 2. Import AQA Overlay ---
try:
    from aqa_overlay import assess_video, warmup
    warmup() # Load the VideoMAE model into memory once at startup
except ImportError as e:
    print(f"WARNING: Could not import aqa_overlay. Error: {e}")
    assess_video = None

# --- 3. Import the New Wage Engine ---
try:
    from wage_engine import predict_wage
except ImportError:
    # Fallback if wage_engine.py is missing
    def predict_wage(skill_score, complexity, demand):
        return "₹150 - ₹200/hr", "85% (Mock)"

# --- 4. Main API Function ---
def get_aqa_result(video_path: str, filename: str = "unknown.mp4"):
    """
    Processes the video and returns the complete JSON contract for the frontend.
    """
    if assess_video is None:
        return {"error": "AQA module not loaded", "score": 0.0}

    try:
        # A. Get AQA Score and Heatmap from your Day 4 script
        result = assess_video(video_path, heatmap_format="data-uri")
        score = float(result.get("score", 0.5))
        heatmap = result.get("heatmap", "")

        # B. Calculate Dynamic Wage based on the ACTUAL AQA score
        # (Complexity and Demand default to 3, but can be passed from frontend later)
        wage_range, wage_confidence = predict_wage(skill_score=score, complexity=3, demand=3)

        # C. Integrity Check (Placeholder for Phase 4 Heuristics)
        # We will wire in the real integrity_checker.py in the next hour
        integrity_status = "PASSED" 

        # D. Return the complete JSON contract
        return {
            "score": round(score, 4),
            "heatmap": heatmap,
            "wage_range": wage_range,
            "wage_confidence": wage_confidence,
            "integrity_status": integrity_status
        }
        
    except Exception as e:
        print(f"Error in get_aqa_result: {e}")
        return {"error": str(e), "score": 0.0, "heatmap": ""}