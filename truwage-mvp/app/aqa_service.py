import os
import json

# Path to Ayansh's JSON file (2 levels up from app/ to reach ROOT/)
JSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mock_assessments.json"))

def load_aqa_mapping():
    """Loads the pre-computed AQA scores and heatmap paths."""
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r') as f:
            return json.load(f)
    return {}

def get_demo_aqa_score(video_filename: str):
    """
    Looks up the exact score and heatmap for the pre-processed demo videos.
    Falls back to a safe default if a random video is uploaded.
    """
    mapping = load_aqa_mapping()
    
    # Handle Dictionary format: {"video.mp4": {"aqa_score": 0.8, "heatmap_path": "..."}}
    if isinstance(mapping, dict) and video_filename in mapping:
        return mapping[video_filename]
        
    # Handle List format: [{"filename": "video.mp4", "aqa_score": 0.8, ...}]
    if isinstance(mapping, list):
        for item in mapping:
            if item.get("filename") == video_filename or item.get("video_name") == video_filename:
                return item

    # FALLBACK: If the uploaded video isn't one of the pre-processed ones
    return {
        "aqa_score": 0.85,
        "heatmap_path": "explain/default/contact_sheet.png" # Fallback path
    }